"""Alert watcher for a modded_train run, for a line-based monitor: reads the run's
train.jsonl, validation.jsonl and checkpoints.jsonl (config.json for the expert count)
and prints one flushed line per alert, `<time> <run> step <n> <kind>: <detail>`.

  health.py RUN                        every alert in the rows so far; exit 1 if any
  health.py RUN --follow [--from-start] keep polling; old rows only set the baselines

RUN is results/pretrain/NAME or NAME. An alert fires when its condition starts (slow
and wait: --persist rows in a row), again every --every rows while it holds, and
re-arms after --clear rows without it:
  nonfinite  NaN/inf CE, aux CE or tok/s    spike    train CE > its EMA + --spike
  slow       tok/s < --slow x median of the last --window settled rows
  wait       sampler_wait (share of wall-clock) > --wait
  restart    steps went back (resumed from a checkpoint; tok/s and wait settle again)
  imbalance  a MoE layer's max/mean expert load > --imbalance (the trainer logs no
             per-expert loads, so no CV)
  starved    a MoE layer with > --starved of its experts under 0.1 x the mean load
  dropped    dropped routes > --dropped      drift    the mean bias-routed share off
             its trailing median by > --drift
  memory     max_memory_gb > --mem           ckpt     a checkpoint save > --ckpt-s
  val        validation CE nonfinite or > best + --val-up
  quiet      no train row for --quiet min (--follow)
  stopped    done.json written (--follow); "finished" when its stop_reason is "steps"
Spike, slow, wait and router alerts wait for --warm rows; slow and wait also skip
--settle rows after every (re)start.
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

PRETRAIN = Path("/home/yimingz3/src/allie/results/pretrain")
FINITE = ("train_ce", "time_ce", "wdl_ce", "tokens_per_second")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


class Tail:
    """New complete lines of a growing file (restarts from 0 if it shrinks)."""

    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, b""

    def lines(self):
        try:
            with open(self.path, "rb") as f:
                if f.seek(0, 2) < self.pos:
                    self.pos, self.buf = 0, b""
                f.seek(self.pos)
                data = f.read()
        except FileNotFoundError:
            return []
        self.pos += len(data)
        *full, self.buf = (self.buf + data).split(b"\n")
        return [x for x in full if x.strip()]


class Watch:
    def __init__(self, a, name, experts):
        self.a, self.name, self.experts, self.loud = a, name, experts, False
        self.ema, self.best, self.step = None, math.inf, None
        self.rows = self.since = 0
        self.tps, self.bias = deque(maxlen=a.window), deque(maxlen=2 * a.window)
        self.on, self.streak, self.alerts = {}, Counter(), 0

    def emit(self, step, kind, msg):
        self.alerts += self.loud
        if self.loud:
            now = time.strftime("%m-%d %H:%M:%S")
            print(f"{now} {self.name} step {step} {kind}: {msg}", flush=True)

    def flag(self, step, kind, cond, msg, persist=1):
        """Fires once cond held `persist` rows in a row, again every --every rows while
        it holds, and re-arms after --clear rows without it."""
        self.streak[kind] = self.streak[kind] + 1 if cond else 0
        if kind not in self.on:
            if self.streak[kind] >= persist:
                self.emit(step, kind, msg)
                self.on[kind] = [0, 0]
            return
        since, gap = self.on[kind]
        since, gap = since + 1, 0 if cond else gap + 1
        if cond and since >= self.a.every:
            self.emit(step, kind, msg)
            since = 0
        self.on[kind] = [since, gap]
        if gap >= self.a.clear:
            del self.on[kind]

    def train(self, r):
        a, s = self.a, r.get("step", 0)
        if self.step is not None and s <= self.step:
            self.emit(s, "restart", f"steps went back {self.step} -> {s}")
            self.since = 0
        self.step, self.rows, self.since = s, self.rows + 1, self.since + 1
        bad = [f"{k}={r[k]}" for k in FINITE if k in r and not finite(r[k])]
        self.flag(s, "nonfinite", bool(bad), " ".join(bad))
        warm, ce = self.rows > a.warm, r.get("train_ce")
        if finite(ce):
            if self.ema is not None:
                msg = f"train CE {ce:.4f} vs EMA {self.ema:.4f}"
                self.flag(s, "spike", warm and ce > self.ema + a.spike, msg)
            self.ema = ce if self.ema is None else self.ema + a.alpha * (ce - self.ema)
        settled = warm and self.since > a.settle
        tps = r.get("tokens_per_second")
        if settled and finite(tps):
            med = statistics.median(self.tps) if len(self.tps) >= 5 else tps
            msg = f"{tps / 1e3:.1f}K tok/s, {tps / med:.2f} x trailing {med / 1e3:.1f}K"
            self.flag(s, "slow", tps < a.slow * med, msg, a.persist)
            self.tps.append(tps)
        if settled and finite(w := r.get("sampler_wait")):
            self.flag(s, "wait", w > a.wait, f"sampler_wait {w:.3f}", a.persist)
        if finite(mem := r.get("max_memory_gb")):
            self.flag(s, "memory", mem > a.mem, f"max allocated {mem:.1f} GB")
        if warm and "moe_imbalance" in r:
            self.router(s, r)

    def router(self, s, r):
        a = self.a
        top = lambda k: max(enumerate(r[k]), key=lambda x: x[1])
        i, imb = top("moe_imbalance")
        n = len(r["moe_imbalance"])
        msg = f"MoE layer {i + 1}/{n} max/mean load {imb:.2f}"
        self.flag(s, "imbalance", imb > a.imbalance, msg)
        if self.experts:
            i, k = top("moe_starved")
            msg = f"MoE layer {i + 1}/{n}: {k:.0f} of {self.experts} experts starved"
            self.flag(s, "starved", k > a.starved * self.experts, msg)
        i, d = top("moe_dropped")
        self.flag(s, "dropped", d > a.dropped, f"MoE layer {i + 1}/{n} drops {d:.3f}")
        b = statistics.mean(r["moe_bias_routed"])
        med = statistics.median(self.bias) if len(self.bias) >= 10 else b
        msg = f"bias-routed share {b:.3f} vs trailing {med:.3f}"
        self.flag(s, "drift", abs(b - med) > a.drift, msg)
        self.bias.append(b)

    def val(self, r):
        ce, s = r.get("move_ce"), r.get("step")
        if not finite(ce):
            self.emit(s, "val", f"validation move CE {ce}")
            return
        if ce > self.best + self.a.val_up:
            self.emit(s, "val", f"validation move CE {ce:.4f} vs best {self.best:.4f}")
        self.best = min(self.best, ce)

    def ckpt(self, r):
        if r.get("seconds", 0) > self.a.ckpt_s:
            self.emit(r.get("step"), "ckpt", f"save took {r['seconds']:.0f} s")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = ap.add_argument
    add("run")
    add("--follow", action="store_true")
    add("--from-start", action="store_true", help="with --follow: alert on old rows")
    add("--poll", type=float, default=30, help="seconds")
    add("--every", type=int, default=40, help="rows between repeats of a lasting alert")
    add("--warm", type=int, default=20, help="rows")
    add("--settle", type=int, default=2, help="rows after a (re)start")
    add("--persist", type=int, default=2, help="rows")
    add("--clear", type=int, default=5, help="rows")
    add("--spike", type=float, default=0.08, help="nats over the EMA")
    add("--alpha", type=float, default=0.1, help="EMA weight of the newest row")
    add("--slow", type=float, default=0.85)
    add("--window", type=int, default=20, help="rows")
    add("--wait", type=float, default=0.05)
    add("--imbalance", type=float, default=6.0)
    add("--starved", type=float, default=0.1, help="share of a layer's experts")
    add("--dropped", type=float, default=0.01)
    add("--drift", type=float, default=0.05)
    add("--mem", type=float, default=44.0, help="GB")
    add("--ckpt-s", type=float, default=900)
    add("--val-up", type=float, default=0.02)
    add("--quiet", type=float, default=15, help="minutes")
    a = ap.parse_args()
    run = Path(a.run) if "/" in a.run else PRETRAIN / a.run
    try:
        moe = json.loads(json.loads((run / "config.json").read_text())["args"]["arch"])
        experts = (moe.get("moe") or [0])[0]
    except (OSError, ValueError, KeyError):
        experts = 0
    w = Watch(a, run.name, experts)
    w.loud = not a.follow or a.from_start
    files = {"train": "train.jsonl", "val": "validation.jsonl"}
    files["ckpt"] = "checkpoints.jsonl"
    tails = {k: Tail(run / f) for k, f in files.items()}
    done = run / "done.json"
    seen = done.stat().st_mtime_ns if done.exists() else 0
    heard, quiet = time.time(), False
    while True:
        for kind, t in tails.items():
            for line in t.lines():
                try:
                    row = json.loads(line)
                except ValueError:
                    w.emit(w.step, "parse", f"bad {files[kind]} line {line[:60]!r}")
                    continue
                getattr(w, kind)(row)
                if kind == "train":
                    heard, quiet = time.time(), False
        if not a.follow:
            sys.exit(1 if w.alerts else 0)
        w.loud = True
        if not quiet and time.time() - heard > a.quiet * 60:
            quiet = True
            w.emit(w.step, "quiet", f"no train row for {a.quiet:g} min")
        if (m := done.stat().st_mtime_ns if done.exists() else 0) != seen:
            seen, d = m, json.loads(done.read_text())
            kind = "finished" if d["stop_reason"] == "steps" else "stopped"
            w.emit(d["step"], kind, f"done.json stop_reason {d['stop_reason']}")
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
