"""bot.toml -> Config. The token never lives here: it comes from LICHESS_TOKEN."""

import tomllib
from dataclasses import dataclass, field, fields

from .engine import Play


@dataclass
class Challenge:
    rated: bool = True
    casual: bool = True
    speeds: tuple = ("bullet", "blitz", "rapid", "classical")
    min_base: int = 30  # seconds
    max_base: int = 10800
    min_increment: int = 0
    max_increment: int = 180
    humans: bool = True
    bots: bool = False
    per_user: int = 1  # concurrent games with one opponent


@dataclass
class Config:
    model: str = "yimingzhang/allie-2.0"  # Hugging Face repo or directory (config.json, weights)
    device: str = "cpu"
    dtype: str = "bfloat16"
    threads: int = 0  # torch CPU threads (0: torch's default)
    experts: int = 0  # routed experts per token (0: all 16; 8 is faster)
    int8: bool | None = None  # int8 block matrices (None: on CPU): half the memory, faster
    url: str = "https://lichess.org"
    max_games: int = 4
    greeting: str = ""
    challenge: Challenge = field(default_factory=Challenge)
    play: Play = field(default_factory=Play)


def build(cls, d):
    names = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    sub = {"challenge": Challenge, "play": Play}
    return cls(**{k: build(sub[k], v) if k in sub else v for k, v in d.items()})


def load(path, overrides=()):
    """overrides: "key=value" or "section.key=value", values in TOML syntax."""
    with open(path, "rb") as f:
        d = tomllib.load(f)
    for o in overrides:
        key, value = o.split("=", 1)
        *sections, key = key.split(".")
        try:
            value = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            pass
        t = d
        for s in sections:
            t = t.setdefault(s, {})
        t[key] = value
    c = build(Config, d)
    p = c.play
    assert p.mode in ("human", "strongest"), p.mode
    assert p.rating == "opponent" or isinstance(p.rating, int), p.rating
    assert p.search in (0, 5, 8, 25, 128), "search: 0 or 5 / 8 / 25 / 128 simulations"
    return c
