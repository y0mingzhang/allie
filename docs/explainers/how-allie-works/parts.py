"""Drawing parts of How Allie Works: expert grids, token chips, a live board, arrows, bars."""

import sys
from pathlib import Path

import numpy as np
from manim import LEFT, RIGHT, UP, FadeIn, FadeOut, Polygon, Rectangle, VGroup, VMobject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402


def rects(boxes, color, opacity=1.0):
    """One mobject of many (x, y, w, h) rectangles, x, y their lower left corners: thousands draw as one path."""
    m = VMobject(fill_color=color, fill_opacity=opacity, stroke_width=0)
    for x, y, w, h in boxes:
        m.start_new_path(np.array([x, y, 0.0]))
        m.add_points_as_corners([np.array(p + (0.0,)) for p in ((x + w, y), (x + w, y + h), (x, y + h), (x, y))])
    return m


class Grid(VGroup):
    """An n x n grid of experts, numbered left to right from the top, unlit cells drawn as one path."""

    def __init__(self, size, n=16, gap=0.14, color=S.PANEL):
        super().__init__()
        self.n, self.pitch = n, size / n
        self.side = self.pitch * (1 - gap)
        self.full = n * self.pitch - (self.pitch - self.side)
        self.base = rects([(*self.corner(e), self.side, self.side) for e in range(n * n)], color)
        self.add(self.base)

    def corner(self, e):
        return (e % self.n) * self.pitch, -(e // self.n) * self.pitch - self.side

    def place(self, e):
        """Scale and lower left corner of expert e's cell at the grid's current position."""
        k = self.base.width / self.full
        o = self.base.get_corner(UP + LEFT)
        x, y = self.corner(int(e))
        return k, o[0] + k * x, o[1] + k * y

    def cells(self, ids, color=S.BRASS, opacity=1.0):
        """A path of these experts' cells at the grid's current position; not added to the grid."""
        boxes = []
        for e in ids:
            k, x, y = self.place(e)
            boxes.append((x, y, k * self.side, k * self.side))
        return rects(boxes, color, opacity)

    def center_of(self, e):
        k, x, y = self.place(e)
        return np.array([x + k * self.side / 2, y + k * self.side / 2, 0.0])


CHIP_H = 0.4


def box(w, h, color=S.PANEL):
    return Rectangle(width=w, height=h, fill_color=color, fill_opacity=1, stroke_width=0)


def chip(s, kind, px=S.SMALL, width=None, h=CHIP_H):
    """A token: White's moves on light chips, Black's on dark, the header on the panel colour."""
    fill, ink = {
        "white": (S.WHITE_SIDE, S.BLACK_SIDE),
        "black": (S.BLACK_SIDE, S.INK),
        "header": (S.PANEL, S.INK),
        "rating": (S.PANEL, S.HILITE),
    }[kind]
    t = S.text(s, px, ink, font=S.MONO)
    r = box(width or max(t.width + 0.18, 0.34), h, fill)
    if kind == "black":
        r.set_stroke(S.RULE, 1.5)
    return VGroup(r, t.move_to(r))


def chips(tokens, kinds, width, gap=0.07, line=0.5, px=S.SMALL, h=CHIP_H):
    """Tokens laid out like text, left to right in lines of at most width. Top left corner at the origin."""
    out, x, y = VGroup(), 0.0, 0.0
    for s, k in zip(tokens, kinds):
        c = chip(s, k, px, h=h)
        if x + c.width > width and x > 0:
            x, y = 0.0, y - line
        c.move_to([x + c.width / 2, y - h / 2, 0])
        x += c.width + gap
        out.add(c)
    return out


def arrow(a, b, width, color=S.BRASS, opacity=0.92, head=2.4, start=0.2, end=0.16):
    """A flat arrow from point a to point b."""
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    u = d / np.linalg.norm(d)
    a, b = a + start * u, b - end * u
    n = np.array([-u[1], u[0], 0.0])
    length = np.linalg.norm(b - a)
    hl = min(max(width * head, 0.2) / 2 * 1.45, 0.6 * length)
    hw = hl / 1.45
    neck = b - hl * u
    w = width / 2
    return Polygon(a + w * n, neck + w * n, neck + hw * n, b, neck - hw * n, neck - w * n, a - w * n,
                   fill_color=color, fill_opacity=opacity, stroke_width=0)  # fmt: skip


def arrows(board, dist, color=S.BRASS, floor=0.03, scale=0.3):
    """Arrows for a {uci: p} distribution on a board, the likeliest drawn last."""
    out = VGroup()
    for uci, p in sorted(dist.items(), key=lambda kv: kv[1]):
        if p >= floor:
            out.add(arrow(board.at(uci[:2]), board.at(uci[2:4]), 0.035 + scale * p**0.8, color))
    return out


def move_bars(rows, width, label_w=1.15, row=0.46, px=S.BODY, color=S.BRASS, highlight=None):
    """rows of (label, p): label, bar and percentage, top left at the origin. highlight: the label to mark."""
    out = VGroup()
    for i, (name, p) in enumerate(rows):
        y = -i * row
        ink = S.INK if name != highlight else S.HILITE
        n = S.text(name, px, ink, font=S.MONO).move_to([label_w / 2, y, 0]).align_to([0, y, 0], LEFT)
        track = box(width, row * 0.56).move_to([label_w + width / 2, y, 0])
        bar = box(max(width * p, 0.01), row * 0.56, color).move_to([label_w + width * p / 2, y, 0])
        val = S.text(f"{100 * p:.0f}%" if p >= 0.01 else "<1%", px, S.INK2, font=S.MONO)
        val.next_to(track, RIGHT, buff=0.14)
        out.add(VGroup(n, track, bar, val))
    return out


def eval_bar(white, draw, black, height, width=0.26):
    """Win, draw and loss chances from White's side as a vertical bar, White at the bottom. Origin: bar's centre."""
    h = [height * x for x in (white, draw, black)]
    y = -height / 2
    out = VGroup()
    for v, c in zip(h, (S.WHITE_SIDE, S.DRAW_SIDE, S.BLACK_SIDE)):
        if v > 0.001:
            out.add(
                Rectangle(width=width, height=v, fill_color=c, fill_opacity=1, stroke_width=0).move_to(
                    [0, y + v / 2, 0]
                )
            )
        y += v
    return out


def clock_text(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def play_move(b, uci):
    """Animations of a legal move on an S.board, keeping b.pieces right."""
    frm, to = uci[:2], uci[2:4]
    p = b.pieces.pop(frm)
    anims = []
    gone = to if to in b.pieces else to[0] + frm[1] if p.symbol in "Pp" and frm[0] != to[0] else None
    if gone:
        anims.append(FadeOut(b.pieces.pop(gone)))
    anims.append(p.animate.move_to(b.at(to)))
    if p.symbol in "Kk" and abs(ord(frm[0]) - ord(to[0])) == 2:
        r0, r1 = ("h" if to[0] == "g" else "a") + frm[1], ("f" if to[0] == "g" else "d") + frm[1]
        rook = b.pieces.pop(r0)
        anims.append(rook.animate.move_to(b.at(r1)))
        b.pieces[r1] = rook
    if len(uci) == 5:
        queen = S.piece(uci[4].upper() if p.symbol.isupper() else uci[4], b.squares[to].width).move_to(b.at(to))
        anims += [FadeOut(p), FadeIn(queen)]
        b.add(queen)
        p = queen
    b.pieces[to] = p
    return anims


def tint(b, squares, color=S.HILITE, opacity=0.75):
    """Colours squares of an S.board under its pieces; returns the board."""
    marks = VGroup(
        *[
            box(b.squares[s].width, b.squares[s].width, color).set_fill(opacity=opacity).move_to(b.at(s))
            for s in squares
        ]
    )
    b.submobjects.insert(1, marks)
    return b


def barcode(values, width, height, scale, levels=6):
    """A vector as columns coloured by value on the diverging ramp, one path per colour; centred at the origin."""
    v = np.asarray(values, float) / scale
    pitch = width / len(v)
    out = VGroup()
    for sign, pole in ((1, S.BRASS), (-1, S.SKY)):
        for lvl in range(1, levels + 1):
            lo, hi = (lvl - 1) / levels, lvl / levels if lvl < levels else np.inf
            ids = np.flatnonzero((sign * v > lo) & (sign * v <= hi))
            if len(ids):
                c = S.mix(S.PANEL, pole, (lvl - 0.25) / levels)
                out.add(rects([(i * pitch - width / 2, -height / 2, pitch, height) for i in ids], c))
    return VGroup(rects([(-width / 2, -height / 2, width, height)], S.PANEL), out)


def dial(radius, angle):
    """A clock hand at an angle, 0 pointing up and turning clockwise like a clock."""
    from manim import Circle, Dot, Line

    face = Circle(radius, stroke_color=S.RULE, stroke_width=2.5, fill_color=S.PANEL, fill_opacity=1)
    tip = radius * 0.86 * np.array([np.sin(angle), np.cos(angle), 0.0])
    hand = Line(np.zeros(3), tip, stroke_color=S.INK, stroke_width=3.5)
    return VGroup(face, hand, Dot(np.zeros(3), radius=0.022, color=S.INK))


def plane(squares, size, color=S.INK, base=S.PANEL):
    """An 8 x 8 piece plane: square indices 0 (a1) to 63 (h8) lit. Lower left corner at the origin."""
    p = size / 8
    s = p * 0.88
    cell = lambda q: ((q % 8) * p, (q // 8) * p, s, s)
    return VGroup(
        rects([cell(q) for q in range(64)], base), *([rects([cell(q) for q in squares], color)] if squares else [])
    )
