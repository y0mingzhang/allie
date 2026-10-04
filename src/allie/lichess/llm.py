"""Claude for the game chat: the Anthropic SDK, streamed, under a spend cap."""

import fcntl
import json
import logging
import math
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

log = logging.getLogger(__name__)

DECISION = {
    "type": "object",
    "properties": {"speak": {"type": "boolean"}, "text": {"type": "string"}},
    "required": ["speak", "text"],
    "additionalProperties": False,
}


class Reply(NamedTuple):
    content: list  # the assistant turn, as returned
    speak: bool
    text: str
    first: float  # seconds to the first token
    seconds: float
    usd: float


MODELS, LOCK = {}, threading.Lock()


def model(cfg):
    """The model of a [chat] config, shared by the games: a function (system, messages) ->
    Reply, or None when it has none (an error, a refusal, the spend cap)."""
    with LOCK:
        if repr(cfg) not in MODELS:
            MODELS[repr(cfg)] = make(cfg)
        return MODELS[repr(cfg)]


def make(cfg):
    if cfg.llm == "mock":
        from .chat import Mock

        return Mock()
    silent = lambda system, messages: None
    if cfg.model not in PRICES:
        log.error("chat: no price for %s, so no spend cap: fixed lines only", cfg.model)
        return silent
    if not (k := key(cfg.key_file)):
        log.error("chat: no ANTHROPIC_API_KEY or %s: fixed lines only", cfg.key_file)
        return silent
    try:
        return Claude(cfg, k)
    except ImportError:
        log.error("chat: no anthropic package (allie[chat]): fixed lines only")
        return silent


def key(path):
    """The Anthropic API key: ANTHROPIC_API_KEY, else the first line of path (chmod 600)."""
    if k := os.environ.get("ANTHROPIC_API_KEY"):
        return k
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            k = f.readline().strip()
    except OSError:
        return None
    if os.stat(path).st_mode & 0o077:
        log.warning("chat: %s is readable by others; chmod 600 it", path)
    return k or None


PRICES = {  # USD per million tokens: input, cache write 5 min and 1 h, cache read, output
    "claude-sonnet-5-5": (2.0, 2.5, 4.0, 0.2, 10.0),
    "claude-sonnet-5": (2.0, 2.5, 4.0, 0.2, 10.0),  # Sonnet 5.5's refusal fallback
    "claude-haiku-4-5": (1.0, 1.25, 2.0, 0.1, 5.0),
}


def cost(model, u):
    """USD of one response's usage."""
    c = getattr(u, "cache_creation", None)
    hour = (getattr(c, "ephemeral_1h_input_tokens", 0) or 0) if c else 0
    written = (u.cache_creation_input_tokens or 0) - hour
    n = (u.input_tokens, written, hour, u.cache_read_input_tokens or 0, u.output_tokens)
    return sum(a * b for a, b in zip(n, PRICES[model])) / 1e6


class Ledger:
    """Spend in USD by UTC day and month ("2026-10-04", "2026-10"), in a JSON file shared by
    processes (read before each check, locked to add). allows() is False from the call that
    reaches a cap until the window ends; for good once the file holds something else than a
    ledger; and for RETRY seconds after the file can't be read or written (a filesystem
    outage), when charges not yet recorded are kept and written with the next one. A missing
    file is a new ledger."""

    RETRY = 300

    def __init__(self, path, day_cap, month_cap, now=lambda: datetime.now(UTC)):
        self.path, self.caps, self.now = (
            os.path.expanduser(path),
            (day_cap, month_cap),
            now,
        )
        self.lock, self.warned, self.broken = threading.Lock(), set(), False
        self.retry, self.unrecorded = None, 0.0  # when to try the file again; USD owed to it

    def read(self):
        try:
            with open(self.path) as f:
                spent = json.load(f)
        except FileNotFoundError:
            return {}
        except OSError as e:
            return self.fail(e, transient=True)
        except ValueError as e:
            return self.fail(e)
        return spent if isinstance(spent, dict) else self.fail("not a JSON object")

    def fail(self, e, transient=False):
        if not self.broken and (self.retry is None or not transient):
            until = "it can be written again" if transient else "this is fixed"
            log.error("chat: ledger %s: %s; no paid calls until %s", self.path, e, until)
        if transient:
            self.retry = self.now() + timedelta(seconds=self.RETRY)
        else:
            self.broken = True

    def windows(self):
        t = self.now()
        return t.strftime("%Y-%m-%d"), t.strftime("%Y-%m")

    def allows(self):
        if self.broken:
            return False
        if (retry := self.retry) is not None:
            if self.now() < retry or math.isinf(self.add(0.0)[0]):
                return False
        spent = self.read()
        if spent is None or self.broken:
            return False
        for w, cap in zip(self.windows(), self.caps):
            if spent.get(w, 0.0) >= cap:
                if w not in self.warned:
                    self.warned.add(w)
                    log.warning("chat: spent $%.2f in %s, the cap: silent until it ends",
                                spent[w], w)  # fmt: skip
                return False
        return True

    def add(self, usd):
        """Charge usd; the window totals (infinite if the charge can't be recorded)."""
        with self.lock:
            self.unrecorded += usd
            if self.broken:
                return [math.inf] * 2
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path + ".lock", "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if (spent := self.read()) is None:
                        return [math.inf] * 2
                    for w in self.windows():
                        spent[w] = spent.get(w, 0.0) + self.unrecorded
                    with open(self.path + ".tmp", "w") as f:
                        json.dump(spent, f, indent=0, sort_keys=True)
                    os.replace(self.path + ".tmp", self.path)
            except OSError as e:
                self.fail(e, transient=True)
                return [math.inf] * 2
            if self.retry is not None:
                log.info("chat: ledger %s written again ($%.4f caught up)", self.path, self.unrecorded)
                self.retry = None
            self.unrecorded = 0.0
            return [spent[w] for w in self.windows()]


class Claude:
    """Claude through the Anthropic SDK, streamed. Errors return None: a bad key, model or
    request turns the model off, a rate limit pauses it a minute. The ledger caps the
    spend; usage keeps the totals."""

    def __init__(self, cfg, key):
        import anthropic

        self.sdk, self.cfg, self.until = anthropic, cfg, 0.0
        self.client = anthropic.Anthropic(
            api_key=key, timeout=cfg.timeout, max_retries=0
        )
        self.ledger = Ledger(cfg.ledger, cfg.day_cap, cfg.month_cap)
        names = ("calls", "new", "cached", "written", "output", "usd")
        self.usage = dict.fromkeys(names, 0)
        output = {"format": {"type": "json_schema", "schema": DECISION}}
        if cfg.effort:
            output["effort"] = cfg.effort
        self.params = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "output_config": output,
            "cache_control": {"type": "ephemeral"},  # the conversation so far
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if cfg.thinking:
            self.params["thinking"] = {"type": cfg.thinking}

    def __call__(self, system, messages):
        a, start = self.sdk, time.monotonic()
        if start < self.until or not self.ledger.allows():
            return None
        first = seen = r = None
        try:
            with self.client.beta.messages.stream(
                system=system, messages=messages, **self.params
            ) as s:
                left = max(self.cfg.timeout - (time.monotonic() - start), 0)
                deadline = threading.Timer(left, s.close)  # the total deadline
                deadline.start()
                try:
                    for ev in s:
                        if ev.type == "message_start":
                            seen = ev.message.usage
                        elif first is None and ev.type == "content_block_delta":
                            first = time.monotonic() - start
                    r = s.get_final_message()
                finally:
                    deadline.cancel()
        except (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError,
                a.BadRequestError) as e:  # fmt: skip
            log.error("chat model off: %s", e)
            self.until = math.inf
        except a.RateLimitError:
            log.warning("chat model rate limited; pausing a minute")
            self.until = start + 60
        except Exception as e:  # noqa: BLE001 - timeouts, the deadline, connection, server
            took = time.monotonic() - start
            log.warning(
                "chat model: %s (%.1f s), message dropped", e or type(e).__name__, took
            )
        if (
            r is None
        ):  # a stream that started is billed: what it read, all it could write
            if seen is not None:
                worst = self.cfg.max_tokens * PRICES[self.cfg.model][4] / 1e6
                self.ledger.add(cost(self.cfg.model, seen) + worst)
            return None
        u, t = r.usage, time.monotonic() - start
        usd = cost(r.model if r.model in PRICES else self.cfg.model, u)
        day, month = self.ledger.add(usd)
        n = (1, u.input_tokens, u.cache_read_input_tokens or 0,
             u.cache_creation_input_tokens or 0, u.output_tokens, usd)  # fmt: skip
        for k, v in zip(self.usage, n):
            self.usage[k] += v
        log.info("chat model %s: %d new + %d cached + %d written, %d out, $%.4f (day $%.2f,"
                 " month $%.2f), first token %.2f s, %.2f s, %s", r.model, *n[1:], day,
                 month, first or t, t, r.stop_reason)  # fmt: skip
        if r.stop_reason != "end_turn" or t > self.cfg.timeout:
            return None
        try:
            d = json.loads("".join(b.text for b in r.content if b.type == "text"))
        except ValueError:
            return None
        return Reply(
            r.content, bool(d.get("speak")), d.get("text", ""), first or t, t, usd
        )
