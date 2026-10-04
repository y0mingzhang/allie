"""Minimal Lichess Bot API client on the standard library: JSON calls and ndjson streams."""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)


class Lichess:
    def __init__(self, token, url="https://lichess.org", timeout=20, retries=5, wait=60):
        self.url, self.timeout, self.retries = url.rstrip("/"), timeout, retries
        self.wait = wait  # after HTTP 429, as Lichess asks
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": "allie-bot"}

    def _open(self, method, path, data=None, timeout=None):
        body = urllib.parse.urlencode(data or {}).encode() if method == "POST" else None
        req = urllib.request.Request(self.url + path, body, self.headers, method=method)
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def call(self, method, path, data=None):
        """JSON response of one call. 429: wait a minute; server and network errors: retry with
        backoff; other HTTP errors raise."""
        for attempt in range(self.retries):
            try:
                with self._open(method, path, data) as r:
                    return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    log.warning("rate limited on %s; waiting %d s", path, self.wait)
                    time.sleep(self.wait)
                    continue
                if e.code < 500:
                    raise
                log.warning("%s %s: HTTP %d, retry %d", method, path, e.code, attempt + 1)
                time.sleep(min(2**attempt, 30))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                log.warning("%s %s failed (%s), retry %d", method, path, e, attempt + 1)
                time.sleep(min(2**attempt, 30))
        raise ConnectionError(f"{method} {path} failed {self.retries} times")

    def stream(self, path, timeout=30):
        """Events of an ndjson stream, None for each keepalive newline; ends when the server
        closes it. Lichess sends keepalives every few seconds, so a silent socket times out."""
        with self._open("GET", path, timeout=timeout) as r:
            for line in r:
                yield json.loads(line) if line.strip() else None

    def account(self):
        return self.call("GET", "/api/account")

    def accept(self, challenge):
        return self.call("POST", f"/api/challenge/{challenge}/accept")

    def decline(self, challenge, reason="generic"):
        return self.call(
            "POST", f"/api/challenge/{challenge}/decline", {"reason": reason}
        )

    def move(self, game, move, draw=False):
        q = "?offeringDraw=true" if draw else ""
        return self.call("POST", f"/api/bot/game/{game}/move/{move}{q}")

    def abort(self, game):
        return self.call("POST", f"/api/bot/game/{game}/abort")

    def resign(self, game):
        return self.call("POST", f"/api/bot/game/{game}/resign")

    def draw(self, game, accept):
        return self.call(
            "POST", f"/api/bot/game/{game}/draw/{'yes' if accept else 'no'}"
        )

    def claim_victory(self, game):
        return self.call("POST", f"/api/bot/game/{game}/claim-victory")

    def chat_lines(self, game):
        return self.call("GET", f"/api/bot/game/{game}/chat")

    def chat(self, game, text, room="player"):
        return self.call(
            "POST", f"/api/bot/game/{game}/chat", {"room": room, "text": text}
        )
