"""Fonts, palette, numbers and captions shared by the animations; text sizes in CSS px at 1280x720."""

import json
import re
from pathlib import Path

import manimpango
from manim import DOWN, LEFT, UP, FadeIn, FadeOut, MarkupText, Text, smooth

HERE = Path(__file__).resolve().parent
for f in sorted((HERE.parent / "fonts").glob("*.ttf")):
    manimpango.register_font(str(f))

# every number a video shows, one top-level key per video
DATA = json.loads((HERE / "data.json").read_text())

# every video's palette and typefaces
BG, INK, INK2, RULE, MUTED = "#F4F1EA", "#16202A", "#46505A", "#8A8F94", "#78716c"
ORANGE, BLUE, DENSE = "#B5501F", "#2F6690", "#5B6B78"
OFF, OFF_EDGE, PANEL = "#FBFAF6", "#C9C2B4", "#EBE6DB"
SANS, MONO = "IBM Plex Sans", "IBM Plex Mono"

PX = 0.72  # manim font_size per CSS pixel
READ_WPS = 3.0  # caption reading speed, words per second


def text(s, px=30, color=INK2, font=SANS, weight="MEDIUM", markup=False, **kw):
    cls = MarkupText if markup else Text
    return cls(s, font=font, font_size=px * PX, color=color, weight=weight, **kw)


def paper(scene, color=BG):
    scene.camera.background_color = color


def kicker(s, color=ORANGE):
    """A section label in the top-left corner."""
    return text(s.upper(), 22, color, weight="SEMIBOLD").to_corner(UP + LEFT, buff=0.45)


class Captions:
    """One caption at a time, centred at the bottom; in step with the beats it names.

    self.play(*cap("the router scores all 256 experts"), other_anim)
    cap.hold()  # wait until the caption has been up long enough to read

    A new caption fades in after the old one has faded out, within the same play. A caption
    wider than WIDE wraps to two lines, growing upward from the one-line position.
    """

    WIDE = 12.6

    def __init__(self, scene, y=-3.42, px=30, color=INK, wps=READ_WPS):
        self.scene, self.y, self.px, self.color, self.wps = scene, y, px, color, wps
        self.cur, self.since, self.words = None, 0.0, 0

    def make(self, s, markup):
        t = text(s, self.px, self.color, markup=markup)
        if t.width > self.WIDE:
            w = re.split(r" (?![^<]*>)", s)  # spaces outside markup tags
            n = len(plain(s))
            cut = min(
                range(1, len(w)), key=lambda i: abs(len(plain(" ".join(w[:i]))) - n / 2)
            )
            s = " ".join(w[:cut]) + "\n" + " ".join(w[cut:])
            t = text(s, self.px, self.color, markup=markup, line_spacing=0.9)
        one = text("Hx", self.px).height
        return t.move_to([0, self.y, 0]).align_to([0, self.y - one / 2, 0], DOWN)

    def __call__(self, s, markup=True):
        out = []
        if self.cur is not None:
            out = [FadeOut(self.cur, shift=0.12 * UP, rate_func=_first_half)]
        self.cur = self.make(s, markup)
        self.since, self.words = self.scene.time, len(plain(s).split())
        return [*out, FadeIn(self.cur, shift=0.12 * UP, rate_func=_second_half)]

    def clear(self):
        if self.cur is None:
            return []
        out, self.cur = FadeOut(self.cur, shift=0.12 * DOWN), None
        return [out]

    def hold(self, extra=0.0, floor=1.6):
        """Wait until the caption has been up words / wps seconds (at least floor), plus extra."""
        left = (
            max(floor, self.words / self.wps) + extra - (self.scene.time - self.since)
        )
        if left > 0.05:
            self.scene.wait(left)


def plain(s):
    return re.sub("<[^>]+>", "", s)


def _first_half(t):
    return smooth(min(1.0, 2 * t))


def _second_half(t):
    return smooth(max(0.0, 2 * t - 1))
