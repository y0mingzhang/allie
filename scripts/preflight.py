"""Node preflight, about a minute: all visible GPUs at once (the power cap binds under
full load) check their memory and a BF16 GEMM against FP64, then time BF16 8192^3 GEMMs
for --seconds while nvidia-smi samples SM clock, power, temperature and clock-event
reasons. Writes one JSON (node, per-GPU fields and verdicts) and exits 1 when a GPU
errors, computes wrongly or runs below --min-frac x the node median or --floor TF/s.
Slow but healthy cards (an L40S held at 1275 MHz by the 350 W cap on babel-q9-16 vs
1410-1485 MHz elsewhere) are only marked slow; a node trains at its slowest GPU's pace.

  preflight.py --out FILE [--seconds 20] [--min-frac 0.7] [--floor 100] [--slow 0.95]
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

FIELDS = ["uuid", "clocks.sm", "clocks.max.sm", "power.draw", "power.limit"]
FIELDS += ["temperature.gpu", "ecc.errors.uncorrected.volatile.total"]
# older drivers name the reasons clocks_throttle_reasons, some lack ECC counters
FIELD_SETS = [FIELDS + ["clocks_event_reasons.active"], FIELDS, FIELDS[:6]]
REASONS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal"}
REASONS |= {0x40: "hw_thermal", 0x80: "hw_power_brake"}


def probe(seconds):
    """One GPU (the only visible one): prints its JSON line."""
    out = {}
    try:
        import torch

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        d = torch.cuda.get_device_properties(0)
        out |= {"name": d.name, "uuid": str(d.uuid), "memory_gb": d.total_memory / 1e9}
        x, y = (torch.randn(2048, 2048, device="cuda").bfloat16() for _ in range(2))
        ref, got = x.double() @ y.double(), x @ y
        out["rel_err"] = ((got.double() - ref).norm() / ref.norm()).item()
        out["repeatable"] = torch.equal(got, x @ y)
        free, _ = torch.cuda.mem_get_info()
        m = torch.empty(int(free * 0.8), dtype=torch.uint8, device="cuda").fill_(165)
        out["memory_ok"] = m.min().item() == m.max().item() == 165
        del m
        n, bf16 = 8192, torch.bfloat16
        x, y = (torch.randn(n, n, device="cuda", dtype=bf16) for _ in range(2))
        for _ in range(10):
            z = x @ y
        torch.cuda.synchronize()
        t0, it = time.time(), 0
        while time.time() - t0 < seconds:
            for _ in range(20):
                z = x @ y
            torch.cuda.synchronize()
            it += 20
        t1 = time.time()
        out |= {"t0": t0, "t1": t1, "tflops": 2 * n**3 * it / (t1 - t0) / 1e12}
        out["finite"] = bool(torch.isfinite(z).all())
    except (RuntimeError, AssertionError, ImportError, OSError) as e:
        out["error"] = repr(e)
    print(json.dumps(out), flush=True)


class Smi:
    """nvidia-smi rows (receive time, {field: value}) every 500 ms, in a thread."""

    def __init__(self):
        self.rows, self.proc, self.done = [], None, False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        for fields in FIELD_SETS:
            cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}"]
            cmd += ["--format=csv,noheader,nounits", "-lms", "500"]
            try:
                kw = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
                self.proc = subprocess.Popen(cmd, **kw, text=True)
            except FileNotFoundError:
                return
            for line in self.proc.stdout:
                vals = [v.strip() for v in line.split(",")]
                if len(vals) == len(fields):
                    self.rows.append((time.time(), dict(zip(fields, vals))))
            if self.rows or self.done:
                return

    def stop(self):
        self.done = True
        if self.proc:
            self.proc.kill()


def column(rows, k):
    return [v for f in rows if (v := num(f.get(k, ""))) is not None]


def num(x):
    try:
        return float(int(x, 16) if x.startswith("0x") else x)
    except ValueError:
        return None


def judge(res, rows, a):
    """Merge each probe with the nvidia-smi samples of its timed window; verdicts."""
    for r in res:
        uuid, t0, t1 = r.get("uuid"), r.get("t0", 0), r.get("t1", 0)
        mine = [f for t, f in rows if t0 <= t <= t1 and f["uuid"].endswith(str(uuid))]
        if clock := column(mine, "clocks.sm"):
            r |= {"sm_mhz": statistics.median(clock), "sm_mhz_min": min(clock)}
        for key, k, fn in (
            ("max_mhz", "clocks.max.sm", max),
            ("power_w", "power.draw", statistics.median),
            ("power_limit_w", "power.limit", max),
            ("temp_c", "temperature.gpu", max),
            ("ecc_uncorrected", "ecc.errors.uncorrected.volatile.total", max),
        ):
            if v := column(mine, k):
                r[key] = fn(v)
        bits = 0
        for v in column(mine, "clocks_event_reasons.active"):
            bits |= int(v)
        r["clock_reasons"] = [name for bit, name in REASONS.items() if bits & bit]
    tf = [r["tflops"] for r in res if "tflops" in r]
    med = statistics.median(tf) if tf else 0.0
    for r in res:
        fail = [r["error"]] if "error" in r else []
        if "tflops" in r:
            t = r["tflops"]
            checks = (
                (r["rel_err"] > 1e-2, f"GEMM rel err {r['rel_err']:.1e}"),
                (not r["repeatable"], "GEMM not repeatable"),
                (not r["memory_ok"], "memory pattern mismatch"),
                (not r["finite"], "nonfinite GEMM"),
                (t < a.floor, f"{t:.0f} TF/s < floor {a.floor:g}"),
                (t < a.min_frac * med, f"{t:.0f} TF/s < {a.min_frac:g} x median"),
            )
            fail += [msg for bad, msg in checks if bad]
        fail += ["uncorrected ECC errors"] * (r.get("ecc_uncorrected", 0) > 0)
        r["fail"], r["slow"] = fail, r.get("tflops", 0) < a.slow * med
    report = {"node": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID")}
    report |= {"time": time.strftime("%F %T"), "seconds": a.seconds, "gpus": res}
    report |= {"median_tflops": med, "min_tflops": min(tf, default=0.0)}
    report["pace"] = report["min_tflops"] / med if med else 0.0
    report["ok"] = bool(res) and not any(r["fail"] for r in res)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--min-frac", type=float, default=0.7)
    ap.add_argument("--floor", type=float, default=100, help="TF/s")
    ap.add_argument("--slow", type=float, default=0.95, help="x node median")
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.probe:
        return probe(a.seconds)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = gpus.split(",") if gpus else [str(i) for i in range(8)]
    smi = Smi()
    cmd = [sys.executable, __file__, "--probe", "--seconds", str(a.seconds)]
    kw = {"stdout": subprocess.PIPE, "text": True}
    procs = [
        subprocess.Popen(cmd, env=os.environ | {"CUDA_VISIBLE_DEVICES": g}, **kw)
        for g in gpus
    ]
    res = []
    for g, p in zip(gpus, procs):
        try:
            out = p.communicate(timeout=a.seconds + 300)[0].strip().splitlines()
            res.append({"gpu": g} | json.loads(out[-1]))
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            p.kill()
            res.append({"gpu": g, "error": f"probe rc {p.poll()}: {e!r}"})
    smi.stop()
    report = judge(res, smi.rows, a)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    tf = sorted(round(r.get("tflops", 0)) for r in res)
    mhz = sorted(round(r["sm_mhz"]) for r in res if "sm_mhz" in r)
    slow = [r["gpu"] for r in res if r["slow"]]
    print(
        f"preflight {report['node']}: {len(res)} GPUs, BF16 TF/s {tf}, SM MHz {mhz}, "
        f"pace {report['pace']:.3f}, slow {slow}, failed "
        f"{ {r['gpu']: r['fail'] for r in res if r['fail']} }",
        flush=True,
    )
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
