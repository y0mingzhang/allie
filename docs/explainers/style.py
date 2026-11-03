"""Allie's visual identity, shared by every explainer video and figure.

The look is the green-and-buff vinyl tournament board with a chess clock's brass on it: flat fills,
square corners, a strict grid. No gradients, glows, shadows, rounded cards or neon. Importing this
file needs neither manim nor matplotlib. From a sibling folder:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import style as S

Surfaces and ink (BOARD theme, the default everywhere; SHEET is the same system for print and PDF)
    GROUND          what everything sits on
    PANEL           raised blocks, unlit cells, bar tracks
    RULE            axes, gridlines, hairlines
    INK INK2 MUTED  primary, secondary, tertiary text (ticks, sources)
    HILITE          the one thing to look at right now; never a chart series

Chess
    SQ_LIGHT SQ_DARK            board squares
    WHITE_SIDE BLACK_SIDE       the two players, and white and black wins in a win/draw/loss bar
    DRAW_SIDE                   draws in that bar

Data. Text wears INK tokens, not series colours: a coloured mark beside a label carries identity.
    CAT             categorical, fixed order, never cycled: BRASS LILAC AQUA FLAG SKY ROSE.
                    Six are safe side by side (lines, bars); only the first four anywhere on a plot
                    (scatter, small multiples). Past that, fold into "other" or facet.
    ALLIE21 ALLIE20 MAIA3 ORIGINAL DENSE
                    colours that follow these entities in every video and figure; DENSE is a neutral
    SEQ             sequential ramp for magnitude, low to high, one hue
    DIV_LOW DIV_MID DIV_HIGH
                    diverging: better than, equal to, worse than the reference

    validate_palette.js on GROUND: CAT adjacent CVD 11.8 / normal 21.9, first four all-pairs 9.4 / 16.9.
    On SHEET: 12.3 / 23.1 and 11.9 / 17.8. Both SEQ ramps pass --ordinal.

Type, in px at 1920x1080. Nothing smaller than SMALL.
    SLAB            Zilla Slab, for titles: words only, since its figures are old-style
    SANS            Fira Sans, for everything else, big numbers included
    MONO            Fira Mono, for tokens, moves, code and columns of digits
    DISPLAY TITLE HEAD CAPTION BODY LABEL SMALL

Layout, in manim units. MARGIN keeps content off the frame edge. CAPTION_TOP: captions own everything
below it. KICKER_Y: the section label's line; content starts below it.

Motion, in seconds. T_QUICK fades and swaps, T_MOVE most moves, T_SLOW a change the viewer must follow,
T_BEAT a pause after a change lands. READ_WPS and READ_FLOOR set how long a caption stays up.

Helpers. mix(a, b, t) blends two colours. Manim: text, kicker, title, tag, ground, piece, board, Captions.
Matplotlib: use_mpl(theme).
"""

import atexit
import functools
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONTS = HERE / "fonts"


@dataclass(frozen=True)
class Theme:
    ground: str
    panel: str
    rule: str
    ink: str
    ink2: str
    muted: str
    hilite: str
    cat: tuple
    dense: str
    seq: tuple
    div: tuple


BOARD = Theme(
    ground="#10291F",
    panel="#17382B",
    rule="#35574A",
    ink="#F3F1DF",
    ink2="#C8D2BF",
    muted="#93A89B",
    hilite="#EBC452",
    cat=("#B88C0E", "#9470D0", "#23A99C", "#C8452F", "#3996DB", "#C8598F"),
    dense="#8A9C90",
    seq=("#566022", "#7E7A1A", "#A89412", "#CFAB1E", "#F0CC55"),
    div=("#3996DB", "#8A9C90", "#C8452F"),
)
SHEET = Theme(
    ground="#F8F9F3",
    panel="#E8EBDF",
    rule="#C4CCBD",
    ink="#10291F",
    ink2="#3C5449",
    muted="#66796E",
    hilite="#8F6C08",
    cat=("#A87F08", "#714CA6", "#009589", "#AF2C1F", "#2686CB", "#AF467E"),
    dense="#6F8076",
    seq=("#C9A93A", "#A87F08", "#7E5F06", "#574204"),
    div=("#2686CB", "#6F8076", "#AF2C1F"),
)

GROUND, PANEL, RULE = BOARD.ground, BOARD.panel, BOARD.rule
INK, INK2, MUTED, HILITE = BOARD.ink, BOARD.ink2, BOARD.muted, BOARD.hilite
CAT = BOARD.cat
BRASS, LILAC, AQUA, FLAG, SKY, ROSE = CAT
ALLIE21, ALLIE20, MAIA3, ORIGINAL, DENSE = BRASS, LILAC, AQUA, FLAG, BOARD.dense
SEQ = BOARD.seq
DIV_LOW, DIV_MID, DIV_HIGH = BOARD.div

SQ_LIGHT, SQ_DARK = "#E4DFC2", "#5C8A6B"
WHITE_SIDE, BLACK_SIDE, DRAW_SIDE = "#FBFAF2", "#0A1410", BOARD.dense

SLAB, SANS, MONO = "Zilla Slab", "Fira Sans", "Fira Mono"
DISPLAY, TITLE, HEAD, CAPTION, BODY, LABEL, SMALL = 132, 76, 52, 40, 34, 28, 24
PX = 0.48  # manim font_size per px at 1920x1080

FRAME_W, FRAME_H = 128 / 9, 8.0
MARGIN = 0.5
CAPTION_TOP = -2.72
KICKER_Y = 3.5

T_QUICK, T_MOVE, T_SLOW, T_BEAT = 0.35, 0.7, 1.4, 0.5
READ_WPS, READ_FLOOR = 2.6, 2.2

_registered = False


def _fonts():
    global _registered
    if not _registered:
        import manimpango

        for f in sorted(FONTS.glob("*.ttf")):
            manimpango.register_font(str(f))
        _registered = True


def mix(a, b, t):
    """The colour a fraction t of the way from hex a to hex b."""
    rgb = [round(int(a[i : i + 2], 16) * (1 - t) + int(b[i : i + 2], 16) * t) for i in (1, 3, 5)]
    return "#{:02X}{:02X}{:02X}".format(*rgb)


@functools.cache
def _piece_dir():
    d = tempfile.mkdtemp(prefix="allie-pieces-")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return Path(d)


def text(s, px=BODY, color=INK, font=SANS, weight="MEDIUM", markup=False, **kw):
    """Text at a px size; markup=True takes Pango spans. Text keeps one submobject per glyph."""
    from manim import MarkupText, Text

    _fonts()
    if markup:
        return MarkupText(s, font=font, font_size=px * PX, color=color, weight=weight, **kw)
    return Text(s, font=font, font_size=px * PX, color=color, weight=weight, disable_ligatures=True, **kw)


def ground(scene, color=GROUND):
    scene.camera.background_color = color


def kicker(s, color=MUTED):
    """The section label, top left."""
    t = text(s.upper(), LABEL, color, weight="SEMIBOLD")
    return t.move_to([-FRAME_W / 2 + MARGIN + t.width / 2, KICKER_Y, 0])


def title(s, px=TITLE, color=INK):
    return text(s, px, color, font=SLAB, weight="SEMIBOLD")


def tag(s, color=INK2, warn=False):
    """Provenance or caveat, always on screen with what it qualifies: tag("measured on Allie 2.0")."""
    from manim import Rectangle, VGroup

    t = text(s, SMALL, color)
    box = Rectangle(width=t.width + 0.34, height=0.46, stroke_color=FLAG if warn else RULE, stroke_width=3)
    return VGroup(box, t.move_to(box))


def piece(sym, size):
    """A piece from its FEN letter, drawn from Colin Burnett's set; size is the square's side."""
    from manim import SVGMobject

    # manim writes a temporary file beside the SVG, so scenes rendering at once must not share a folder
    src = FONTS / "pieces" / f"{sym}.svg"
    tmp = _piece_dir() / src.name
    if not tmp.exists():
        shutil.copy(src, tmp)
    p = SVGMobject(str(tmp), height=size * 0.92)
    for m in p.family_members_with_points():
        m.set_stroke(width=m.get_stroke_width() * size * 0.92 / 0.45)
    p.symbol = sym
    return p


def board(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", size=4.8, flip=False, coords=True):
    """A board from a FEN's placement field. Returns a VGroup with .squares and .pieces keyed by "e4",
    and .at(name) for a square's centre."""
    from manim import LEFT, Square, VGroup

    side = size / 8
    squares, pieces = {}, {}
    placement = fen.split()[0].split("/")
    grid, labels = VGroup(), VGroup()
    for r in range(8):
        for f in range(8):
            name = "abcdefgh"[f] + str(r + 1)
            x, y = (7 - f, 7 - r) if flip else (f, r)
            sq = Square(side, stroke_width=0, fill_opacity=1, fill_color=SQ_DARK if (f + r) % 2 == 0 else SQ_LIGHT)
            squares[name] = sq.move_to([(x - 3.5) * side, (y - 3.5) * side, 0])
            grid.add(sq)
        for rank_row, row in enumerate(placement):
            if rank_row != 7 - r:
                continue
            f = 0
            for c in row:
                if c.isdigit():
                    f += int(c)
                else:
                    name = "abcdefgh"[f] + str(r + 1)
                    pieces[name] = piece(c, side).move_to(squares[name])
                    f += 1
    out = VGroup(grid)
    if coords:
        for i in range(8):
            f, r = "abcdefgh"[7 - i if flip else i], str(8 - i if flip else i + 1)
            labels.add(text(f, SMALL, MUTED).move_to([(i - 3.5) * side, -size / 2 - 0.2, 0]))
            labels.add(text(r, SMALL, MUTED).next_to(grid, LEFT, buff=0.12).set_y((i - 3.5) * side))
        out.add(labels)
    out.add(*pieces.values())
    out.squares, out.pieces, out.grid, out.coords = squares, pieces, grid, labels
    out.at = lambda name: squares[name].get_center()
    return out


class Captions:
    """One caption at a time, left-aligned in the band below CAPTION_TOP. No narration exists: these are it.

    cap = Captions(self)
    cap.say("The router scores all 256 experts", FadeIn(grid))  # plays, then holds until readable
    self.play(*cap("Now the next move"), other_anim); cap.hold()  # same thing, by hand

    At most two lines, or ValueError: shorten the caption. Markup spans are allowed.
    """

    WIDTH = 11.6

    def __init__(self, scene, wps=READ_WPS, floor=READ_FLOOR):
        self.scene, self.wps, self.floor = scene, wps, floor
        self.cur, self.since, self.words = None, 0.0, 0

    def make(self, s):
        import re

        from manim import DOWN, LEFT

        plain = lambda x: re.sub("<[^>]+>", "", x)
        t = text(s, CAPTION, INK, markup=True, line_spacing=0.9)
        if t.width > self.WIDTH and "\n" not in s:
            w = re.split(r" (?![^<]*>)", s)
            n = len(plain(s))
            cut = min(range(1, len(w)), key=lambda i: abs(len(plain(" ".join(w[:i]))) - n / 2))
            s = " ".join(w[:cut]) + "\n" + " ".join(w[cut:])
            t = text(s, CAPTION, INK, markup=True, line_spacing=0.9)
        if t.width > self.WIDTH or s.count("\n") > 1:
            raise ValueError(f"caption over two lines: {plain(s)!r}")
        self.words = len(plain(s).split())
        # one baseline for every caption: a last line with no descenders sits a descender higher
        drop = text("Hg", CAPTION).height - text("H", CAPTION).height
        lift = 0 if set(plain(s).split("\n")[-1]) & set("gjpqyQ,;()[]") else drop
        return t.align_to([-FRAME_W / 2 + MARGIN, 0, 0], LEFT).align_to([0, -FRAME_H / 2 + 0.34 + lift, 0], DOWN)

    def __call__(self, s):
        from manim import FadeIn, FadeOut, smooth

        out = []
        if self.cur is not None:
            out = [FadeOut(self.cur, rate_func=lambda t: smooth(min(1.0, 2 * t)))]
        self.cur = self.make(s)
        self.since = self.scene.time
        return [*out, FadeIn(self.cur, rate_func=lambda t: smooth(max(0.0, 2 * t - 1)))]

    def clear(self):
        from manim import FadeOut

        if self.cur is None:
            return []
        out, self.cur = FadeOut(self.cur), None
        return [out]

    def hold(self, extra=0.0):
        """Wait until the caption has been up words / wps seconds (at least floor), plus extra."""
        left = max(self.floor, self.words / self.wps) + extra - (self.scene.time - self.since)
        if left > 0.05:
            self.scene.wait(left)

    def say(self, s, *anims, run_time=T_MOVE, extra=0.0):
        self.scene.play(*self(s), *anims, run_time=run_time)
        self.hold(extra)


def use_mpl(theme=BOARD):
    """Fonts, colours and marks for a matplotlib figure in this identity."""
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib import font_manager

    for f in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update(
        {
            "font.family": SANS,
            "font.weight": "medium",
            "font.size": 13,
            "text.color": theme.ink,
            "figure.facecolor": theme.ground,
            "savefig.facecolor": theme.ground,
            "axes.facecolor": theme.ground,
            "axes.edgecolor": theme.rule,
            "axes.labelcolor": theme.ink2,
            "axes.titlecolor": theme.ink,
            "axes.prop_cycle": cycler(color=theme.cat),
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": theme.rule,
            "grid.linewidth": 0.6,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
        }
    )
