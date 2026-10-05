"""How Allie Works: a Lichess game becomes tokens, the board and clocks join each move, 24 blocks
route each move through 16 of 256 experts, and three heads predict a human's move, think time and
result; then what "human-like" means, why the experts are small, the router's fixes, active
against total parameters, and Allie 2.1. Every number is data.json's "arch" key."""

import numpy as np
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    ArcBetweenPoints,
    Arrow,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    FunctionGraph,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    ManimColor,
    Polygon,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Square,
    Transform,
    TransformFromCopy,
    TransformMatchingShapes,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    interpolate_color,
    linear,
)

from style import (
    BG,
    BLUE,
    DATA,
    DENSE,
    INK,
    INK2,
    MONO,
    MUTED,
    OFF,
    OFF_EDGE,
    ORANGE,
    PANEL,
    RULE,
    Captions,
    kicker,
    paper,
    text,
)

D = DATA["arch"]
G, V0, V1, R = D["game"], D["v20"], D["v21"], D["router"]
E, K = D["experts"], D["topk"]
SIDE = int(round(E**0.5))
WPS = (
    2.6
)  # caption reading speed: slower than the default, the pictures need watching too
PALE, BLUE_PALE, GREY = "#E9C3AE", "#BFD0DE", "#A9B0B6"
LIGHT_SQ, DARK_SQ = "#EFE8DA", "#CDBFA6"
GLYPH = dict(k="♚", q="♛", r="♜", b="♝", n="♞", p="♟")

VIDEO = dict(
    name="how-allie-works",
    scenes=[
        "Title",
        "Tokens",
        "Inputs",
        "Trunk",
        "Experts",
        "Heads",
        "Human",
        "Small",
        "Router",
        "Params",
        "Next",
    ],
    gif=(["Experts", 3.3], 14),
    poster=["Title", 4.4],
)


def sans(s, px=30, color=INK2, weight="MEDIUM", **kw):
    return text(s, px, color, weight=weight, **kw)


def bold(s, px=30, color=INK):
    return text(s, px, color, weight="SEMIBOLD")


def mono(s, px=28, color=INK):
    return text(s, px, color, font=MONO, weight="NORMAL")


def pct(p):
    return f"{100 * p:.0f}%" if p >= 0.0095 else f"{100 * p:.1f}%"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    m = m.removesuffix(".0")
    return f"{'' if m == '1' else m + '×'}10<sup>{int(e)}</sup>".replace("-", "−")


def sep(px):
    t = sans("·", px, MUTED)
    t.is_sep = True
    return t


def words(*parts, px=36):
    """One line of words on a common baseline; sep() dots sit at mid x-height."""
    line = VGroup(*parts).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
    mid = parts[0].get_bottom()[1] + text("x", px).height / 2
    for p in parts:
        if getattr(p, "is_sep", False):
            p.set_y(mid)
    return line


CHIP = dict(
    start=(ORANGE, BG, ORANGE),
    tc=(BLUE, BG, BLUE),
    elo=(INK, BG, INK),
    move=(OFF, INK, RULE),
)


def chip(s, kind="move", px=28):
    fill, ink, edge = CHIP[kind]
    t = mono(s, px, ink)
    u = px / 28
    box = RoundedRectangle(
        corner_radius=0.07 * u,
        width=max(t.width + 0.24 * u, 0.44 * u),
        height=0.6 * u,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=edge,
        stroke_width=2,
    )
    t.move_to(box)
    if any(c in s for c in "gjpqy"):  # keep the baseline of chips with descenders
        t.shift(DOWN * (mono("xg", px).height - mono("x", px).height) / 2)
    return VGroup(box, t)


def bracket(group, label, px=24, color=INK2):
    lo, hi = group.get_left()[0], group.get_right()[0]
    y = group.get_bottom()[1] - 0.14
    lines = VGroup(
        Line([lo, y, 0], [hi, y, 0]),
        Line([lo, y, 0], [lo, y + 0.1, 0]),
        Line([hi, y, 0], [hi, y + 0.1, 0]),
    ).set_stroke(RULE, 2)
    return VGroup(lines, sans(label, px, color).next_to(lines, DOWN, buff=0.12))


def vec(n=8, color=INK2, seed=0, cell=0.17, vertical=False):
    a = np.random.default_rng(seed).uniform(0, 0.75, n)
    cells = VGroup(
        *[
            Square(
                cell,
                fill_color=interpolate_color(ManimColor(color), ManimColor(BG), x),
                fill_opacity=1,
                stroke_width=0,
            )
            for x in a
        ]
    )
    return cells.arrange(DOWN if vertical else RIGHT, buff=0.03)


def arrow(a, b, color=RULE, buff=0.08, width=3, tip=0.14):
    return Arrow(
        a,
        b,
        buff=buff,
        color=color,
        stroke_width=width,
        tip_length=tip,
        max_tip_length_to_length_ratio=0.5,
        max_stroke_width_to_length_ratio=100,
    )


def plus(r=0.22, at=(0, 0, 0)):
    c = Circle(
        radius=r, fill_color=OFF, fill_opacity=1, stroke_color=INK2, stroke_width=2.5
    )
    p = VGroup(
        Line(LEFT * r * 0.55, RIGHT * r * 0.55), Line(UP * r * 0.55, DOWN * r * 0.55)
    ).set_stroke(INK2, 2.5)
    return VGroup(c, p).move_to(at)


def box(w, h, label, fill, ink=BG, px=26, at=(0, 0, 0)):
    b = RoundedRectangle(
        corner_radius=0.08,
        width=w,
        height=h,
        fill_color=fill,
        fill_opacity=1,
        stroke_width=0,
    ).move_to(at)
    return VGroup(b, sans(label, px, ink).move_to(b))


def grid(cell=0.2, buff=0.045):
    return VGroup(
        *[
            Square(
                cell,
                fill_color=OFF,
                fill_opacity=1,
                stroke_color=OFF_EDGE,
                stroke_width=1.5,
            )
            for _ in range(E)
        ]
    ).arrange_in_grid(SIDE, SIDE, buff=buff)


def shared_bar(g, h=0.46, buff=0.2):
    b = RoundedRectangle(
        corner_radius=0.08,
        width=g.width,
        height=h,
        fill_color=OFF,
        fill_opacity=1,
        stroke_color=BLUE,
        stroke_width=2.5,
    ).next_to(g, UP, buff=buff)
    return VGroup(b, sans("shared expert", 24, BLUE).move_to(b))


def route(seed):
    """Sigmoid scores of the 256 experts for one move, its top 16, and their gates (sum 4)."""
    z = np.random.default_rng(seed).normal(-1.0, 1.6, E)
    s = 1 / (1 + np.exp(-z))
    top = np.argsort(-s)[:K]
    return s, top, s[top] * D["gate_sum"] / s[top].sum()


def shade(s):
    return interpolate_color(ManimColor(OFF), ManimColor(DENSE), float(s))


def light(g, on, edge=OFF_EDGE):
    return [
        c.animate.set_fill(ORANGE if i in on else OFF).set_stroke(
            ORANGE if i in on else edge
        )
        for i, c in enumerate(g)
    ]


def board(fen, sq=0.3):
    squares, pieces = VGroup(), VGroup()
    for r in range(8):
        for f in range(8):
            squares.add(
                Square(
                    sq,
                    fill_color=LIGHT_SQ if (r + f) % 2 == 0 else DARK_SQ,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([(f - 3.5) * sq, (3.5 - r) * sq, 0])
            )
    for r, row in enumerate(fen.split("/")):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
                continue
            g = text(GLYPH[c.lower()], 40, INK, font="DejaVu Sans", weight="NORMAL")
            g.scale_to_fit_height(sq * 0.8)
            if c.isupper():
                g.set_fill(OFF, 1).set_stroke(INK, 1.4)
            pieces.add(g.move_to([(f - 3.5) * sq, (3.5 - r) * sq, 0]))
            f += 1
    frame = Square(8 * sq, stroke_color=OFF_EDGE, stroke_width=2)
    return VGroup(squares, frame, pieces)


def at(b, square):
    """Centre of a square ("e4") on a board() mobject."""
    sq = b[0][0].width
    f, r = ord(square[0]) - 97, int(square[1]) - 1
    return b[0].get_center() + np.array([(f - 3.5) * sq, (r - 3.5) * sq, 0])


def check(ok, s=0.22):
    if ok:
        m = VMobject().set_points_as_corners(
            [[-s, 0, 0], [-s / 3, -s * 0.7, 0], [s, s * 0.8, 0]]
        )
        return m.set_stroke(BLUE, 6)
    return VGroup(
        Line([-s, -s, 0], [s, s, 0]), Line([-s, s, 0], [s, -s, 0])
    ).set_stroke(ORANGE, 6)


def hbars(rows, x0, y0, scale, step=0.62, h=0.38, px=26, fmt=pct, name_px=None):
    """Horizontal bars: rows of (name, value, color, name color), names right-aligned at x0."""
    g = VGroup()
    for j, (name, v, col, ncol) in enumerate(rows):
        y = y0 - step * j
        bar = Rectangle(
            width=max(scale * v, 0.02),
            height=h,
            fill_color=col,
            fill_opacity=1,
            stroke_width=0,
        ).move_to([x0, y, 0], aligned_edge=LEFT)
        g.add(
            VGroup(
                mono(name, name_px or px, ncol).move_to(
                    [x0 - 0.25, y, 0], aligned_edge=RIGHT
                ),
                bar,
                sans(fmt(v), px - 2, INK2).next_to(bar, RIGHT, buff=0.15),
            )
        )
    return g


def fade_all(scene, cap, t=0.7):
    scene.play(*cap.clear(), *[FadeOut(m) for m in scene.mobjects], run_time=t)


class Title(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        t = text("How Allie Works", 92, INK, weight="SEMIBOLD")
        sub = sans("a chess model that plays like a human", 38)
        head = VGroup(t, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        head.move_to([-2.6, 0.45, 0])
        g = grid(0.17, 0.042).move_to([4.4, -0.05, 0])
        sh = shared_bar(g, 0.42, 0.16)
        s, top, _ = route(3)
        self.play(FadeIn(head, shift=0.2 * UP), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in g], lag_ratio=0.004),
            FadeIn(sh),
            *cap("Allie reads a chess game and predicts what a person will do next."),
            run_time=1.4,
        )
        self.play(
            *[
                c.animate.set_fill(shade(s[i])).set_stroke(shade(s[i]))
                for i, c in enumerate(g)
            ],
            run_time=0.8,
        )
        self.play(
            *light(g, set(top.tolist())),
            sh[0].animate.set_fill(BLUE),
            sh[1].animate.set_color(BG),
            run_time=0.7,
        )
        cap.hold(0.8)
        fade_all(self, cap)


class Tokens(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("1 · a game becomes tokens")
        tc, w, b = G["tc"], str(G["white"]), str(G["black"])
        px = 36
        line1 = words(sans("Rated blitz", px, MUTED), sep(px), mono(tc, px))
        line2 = words(
            sans("White", px, INK),
            mono(w, px),
            sep(px),
            sans("Black", px, INK),
            mono(b, px),
        )
        san = []
        for i, m in enumerate(G["san"]):
            if i % 2 == 0:
                san.append(sans(f"{i // 2 + 1}.", px, MUTED))
            san.append(mono(m, px))
        line3 = words(*san)
        lines = VGroup(line1, line2, line3).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        card = RoundedRectangle(
            corner_radius=0.15,
            width=lines.width + 1.0,
            height=lines.height + 0.8,
            fill_color=OFF,
            fill_opacity=1,
            stroke_color=OFF_EDGE,
            stroke_width=2,
        ).move_to([0, 1.55, 0])
        lines.move_to(card)
        self.play(
            FadeIn(k),
            FadeIn(card),
            FadeIn(lines, shift=0.1 * DOWN),
            *cap(
                "On Lichess, a game is a time control, two ratings and a list of moves."
            ),
            run_time=1.0,
        )
        cap.hold(0.3)

        start = chip("START", "start", 30)
        tcs = [chip(str(G["base"]), "tc", 30), chip(str(G["inc"]), "tc", 30)]
        hw = [chip(c, "elo", 30) for c in w]
        hb = [chip(c, "elo", 30) for c in b]
        moves = [chip(m, "move", 30) for m in G["uci"]]
        row = VGroup(start, *tcs, *hw, *hb, *moves).arrange(RIGHT, buff=0.08)
        for c in moves:
            c.shift(0.3 * RIGHT)
        row.move_to([0, -1.0, 0])
        groups = [VGroup(start), VGroup(*tcs), VGroup(*hw), VGroup(*hb)]
        subs = VGroup(
            *[
                sans(n, 24, MUTED).next_to(gr, UP, buff=0.16)
                for gr, n in zip(groups, ["start", "time control", "White", "Black"])
            ]
        )
        header = VGroup(start, *tcs, *hw, *hb)
        hbr = bracket(header, f"header · {D['header_tokens']} tokens", 26, INK)
        self.play(
            FadeIn(start, shift=0.2 * DOWN),
            FadeTransform(line1[2].copy(), VGroup(*tcs)),
            FadeTransform(line2[1].copy(), VGroup(*hw)),
            FadeTransform(line2[4].copy(), VGroup(*hb)),
            FadeIn(subs),
            *cap(
                f"The header is {D['header_tokens']} tokens: start, base time, "
                f"increment, and each rating as {len(w)} digits."
            ),
            run_time=1.3,
        )
        self.play(FadeIn(hbr), run_time=0.5)
        cap.hold()
        msrc = [m for i, m in enumerate(line3) if i % 3]  # 1. e4 e5 2. Nf3 Nc6
        mbr = bracket(VGroup(*moves), "one token per move", 26, INK)
        self.play(
            *[FadeTransform(s.copy(), m) for s, m in zip(msrc, moves)],
            *cap(
                "Then one token per move, from-square to-square: "
                f"{D['move_tokens']:,} possible moves."
            ),
            run_time=1.1,
        )
        self.play(FadeIn(mbr), run_time=0.5)
        cap.hold()

        # many games share one training row; attention stays inside each game
        lens = [len(row), 22, 9, 30, 14, 26, 17]
        x0, width, y = -6.0, 12.0, 2.0
        xs = x0 + width * np.r_[0, np.cumsum(lens)] / sum(lens)
        colors = [ORANGE] + [PALE if i % 2 else BLUE_PALE for i in range(1, len(lens))]
        segs = VGroup(
            *[
                Rectangle(
                    width=b_ - a - 0.04,
                    height=0.42,
                    fill_color=c,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([(a + b_) / 2, y, 0])
                for a, b_, c in zip(xs, xs[1:], colors)
            ]
        )
        rlab = sans(f"one training row: {D['row_tokens']:,} tokens, many games", 26)
        rlab.next_to(segs, UP, buff=0.2)
        self.play(FadeOut(VGroup(card, lines, subs, hbr, mbr)), run_time=0.5)
        self.play(
            row.animate.scale_to_fit_width(segs[0].width).move_to(segs[0]),
            run_time=0.8,
        )
        self.play(
            FadeTransform(row, segs[0]),
            LaggedStart(
                *[FadeIn(s, shift=0.1 * RIGHT) for s in segs[1:]], lag_ratio=0.15
            ),
            FadeIn(rlab),
            run_time=0.9,
        )
        side, top = 3.0, 1.2
        sc = side / width
        frame = Square(
            side, stroke_color=RULE, stroke_width=2, fill_color=OFF, fill_opacity=1
        ).move_to([0, top - side / 2, 0])
        mx = -side / 2
        tris = VGroup(
            *[
                Polygon(
                    [mx + (a - x0) * sc, top - (a - x0) * sc, 0],
                    [mx + (a - x0) * sc, top - (b_ - x0) * sc, 0],
                    [mx + (b_ - x0) * sc, top - (b_ - x0) * sc, 0],
                    fill_color=c,
                    fill_opacity=1,
                    stroke_width=0,
                )
                for a, b_, c in zip(xs, xs[1:], colors)
            ]
        )
        mlab = (
            VGroup(
                sans("which earlier tokens", 26, INK2),
                sans("each token can see", 26, INK2),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            .next_to(frame, RIGHT, buff=0.45)
            .align_to(frame, UP)
        )
        self.play(
            FadeIn(frame),
            LaggedStart(*[FadeIn(t) for t in tris], lag_ratio=0.2),
            FadeIn(mlab),
            *cap(
                f"Training packs games into rows of {D['row_tokens']:,} tokens; "
                "attention stays inside each game."
            ),
            run_time=1.4,
        )
        cap.hold(0.4)
        fade_all(self, cap)


class Inputs(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("2 · what Allie sees at every move")
        xs = [-4.75, 0.0, 4.75]
        titles = [bold(s, 32) for s in ("Move token", "Board", "Clocks")]
        for t, x in zip(titles, xs):
            t.move_to([x, 2.85, 0])
        vy = -1.6
        vecs = [vec(8, INK2, s).move_to([x, vy, 0]) for s, x in zip((1, 2, 3), xs)]

        tok = chip(G["uci"][2], px=40).move_to([xs[0], 1.3, 0])
        emb = box(2.8, 0.66, "embedding table", DENSE, px=26, at=[xs[0], -0.35, 0])
        a1 = arrow(tok.get_bottom(), emb.get_top())
        a2 = arrow(emb.get_bottom(), vecs[0].get_top())
        self.play(
            FadeIn(k),
            FadeIn(titles[0]),
            FadeIn(tok, shift=0.2 * DOWN),
            *cap("Each move token is looked up as a learned vector."),
            run_time=0.9,
        )
        self.play(Create(a1), FadeIn(emb), run_time=0.5)
        self.play(Create(a2), FadeIn(vecs[0], shift=0.15 * DOWN), run_time=0.5)
        cap.hold()

        bd = board(D["demo"]["outputs"]["fen"], 0.29).move_to([xs[1], 1.3, 0])
        maps = VGroup(
            *[
                Square(
                    0.62,
                    fill_color=interpolate_color(
                        ManimColor(BLUE), ManimColor(BG), 0.25 + 0.2 * i
                    ),
                    fill_opacity=1,
                    stroke_color=BG,
                    stroke_width=2,
                ).shift(0.16 * i * RIGHT + 0.1 * i * DOWN)
                for i in range(D["board"]["convs"])
            ]
        )
        cb = D["board"]
        cnn_lab = sans(f"{cb['convs']} conv layers,\n{cb['channels']} channels", 22)
        cnn = VGroup(maps, cnn_lab.next_to(maps, RIGHT, buff=0.2)).move_to(
            [xs[1], -0.4, 0]
        )
        b1 = arrow(bd.get_bottom(), [xs[1], cnn.get_top()[1], 0])
        b2 = arrow([xs[1], cnn.get_bottom()[1], 0], vecs[1].get_top())
        self.play(
            FadeIn(titles[1]),
            FadeIn(bd[:2]),
            LaggedStart(*[FadeIn(p, scale=0.7) for p in bd[2]], lag_ratio=0.03),
            *cap(
                f"It also sees the board after that move: {cb['squares']} squares, "
                f"each empty or one of {cb['states'] - 1} pieces."
            ),
            run_time=1.2,
        )
        cap.hold(-0.6)
        self.play(
            Create(b1),
            FadeIn(cnn),
            *cap("A small convolutional network reads it."),
            run_time=0.7,
        )
        self.play(Create(b2), FadeIn(vecs[1], shift=0.15 * DOWN), run_time=0.5)
        cap.hold()

        c = D["demo"]["clock"]
        rows = VGroup(
            *[
                VGroup(sans(lab, 26), mono(val, 30)).arrange(RIGHT, buff=0.25)
                for lab, val in (
                    ("my time", f"{c['mine'] // 60}:{c['mine'] % 60:02d}"),
                    ("their time", f"{c['theirs'] // 60}:{c['theirs'] % 60:02d}"),
                    ("my last think", "—" if c["think"] is None else f"{c['think']} s"),
                )
            ]
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.16)
        rows.move_to([xs[2], 1.5, 0])
        wx, nw = 1.3, 3
        x_at = -wx + 2 * wx * np.log1p(c["mine"]) / 10

        def wave(i):
            f = 2**i * PI / wx
            return f, 0.3 - 0.3 * i

        waves = VGroup(
            *[
                FunctionGraph(
                    lambda x, f=wave(i)[0]: 0.12 * np.sin(f * x),
                    x_range=[-wx, wx, 0.01],
                    color=BLUE,
                    stroke_width=3,
                ).shift(wave(i)[1] * UP)
                for i in range(nw)
            ]
        )
        probe = DashedLine(
            [x_at, 0.5, 0], [x_at, -0.8, 0], color=ORANGE, stroke_width=3
        )
        dots = VGroup(
            *[
                Dot(
                    [x_at, wave(i)[1] + 0.12 * np.sin(wave(i)[0] * x_at), 0],
                    0.06,
                    color=ORANGE,
                )
                for i in range(nw)
            ]
        )
        fourier = VGroup(waves, probe, dots).move_to([xs[2], -0.3, 0])
        cw = D["clock"]["waves"]
        flab = sans(f"{cw} sines + {cw} cosines of log seconds", 20, MUTED)
        flab.next_to(fourier, DOWN, buff=0.06)
        c1 = arrow(rows.get_bottom(), [xs[2], fourier.get_top()[1], 0])
        c2 = arrow([xs[2], flab.get_bottom()[1], 0], vecs[2].get_top())
        self.play(
            FadeIn(titles[2]),
            FadeIn(rows, shift=0.1 * DOWN),
            *cap(
                "And the clocks: its time left, the opponent's, and its last think time."
            ),
            run_time=0.9,
        )
        cap.hold(-0.5)
        self.play(Create(c1), Create(waves), run_time=0.8)
        self.play(
            Create(probe),
            FadeIn(dots),
            FadeIn(flab),
            *cap("Each time becomes waves of log seconds, so nearby times look alike."),
            run_time=0.7,
        )
        self.play(Create(c2), FadeIn(vecs[2], shift=0.15 * DOWN), run_time=0.5)
        cap.hold()

        p = plus(0.24, [0, -2.5, 0])
        x0 = vec(8, ORANGE, 9).next_to(p, RIGHT, buff=0.5)
        x0lab = sans("one vector per move", 24).next_to(x0, RIGHT, buff=0.25)
        ins = VGroup(
            *[arrow(v.get_bottom(), p.get_center(), buff=0.28, width=2.5) for v in vecs]
        )
        self.play(
            Create(ins),
            FadeIn(p),
            *cap(
                "The three are added into one vector per move. "
                f"{D['clock']['dropped']:.0%} of training games hide the clock."
            ),
            run_time=0.8,
        )
        self.play(FadeIn(x0, shift=0.2 * RIGHT), FadeIn(x0lab), run_time=0.5)
        cap.hold()
        fade_all(self, cap)


class Trunk(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("3 · the trunk")
        n = D["blocks"]
        xs = np.linspace(-6.0, 6.0, n)
        dense = set(V0["dense"])
        yb, h_att, h_mlp = -0.85, 0.55, 1.45
        ys = yb + h_mlp / 2
        blocks = VGroup()
        for i, x in enumerate(xs, 1):
            mlp = RoundedRectangle(
                corner_radius=0.05,
                width=0.38,
                height=h_mlp,
                fill_color=GREY,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([x, yb + h_mlp / 2, 0])
            att = RoundedRectangle(
                corner_radius=0.05,
                width=0.38,
                height=h_att,
                fill_color=DENSE,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([x, yb + h_mlp + 0.06 + h_att / 2, 0])
            num = sans(str(i), 22, MUTED).move_to([x, yb - 0.25, 0])
            blocks.add(VGroup(att, mlp, num))
        blocks.set_z_index(1)
        stream = Line([-6.9, ys, 0], [6.9, ys, 0], stroke_color=RULE, stroke_width=3)
        self.play(
            FadeIn(k),
            Create(stream),
            LaggedStart(*[FadeIn(b, shift=0.15 * UP) for b in blocks], lag_ratio=0.05),
            *cap(
                f"The vectors then pass through {n} transformer blocks, "
                "each attention followed by an MLP."
            ),
            run_time=1.6,
        )
        dot = Dot([-6.9, ys, 0], 0.1, color=ORANGE).set_z_index(2)
        self.play(FadeIn(dot), run_time=0.2)
        self.play(dot.animate.move_to([6.9, ys, 0]), run_time=2.0, rate_func=linear)
        self.play(FadeOut(dot), run_time=0.2)
        cap.hold(-0.4)

        toks = (
            VGroup(
                chip("START", "start", 20),
                chip(str(G["base"]), "tc", 20),
                chip(str(G["inc"]), "tc", 20),
                *[chip(c, "elo", 20) for c in str(G["white"]) + str(G["black"])],
                *[chip(m, "move", 20) for m in G["uci"]],
            )
            .arrange(RIGHT, buff=0.05)
            .move_to([0.4, 2.75, 0])
        )
        last = toks[-1]
        arcs = VGroup(
            *[
                ArcBetweenPoints(
                    last.get_bottom() + 0.03 * DOWN,
                    t.get_bottom() + 0.03 * DOWN,
                    angle=-PI * 0.28,
                ).set_stroke(ORANGE, 2.5, opacity=0.85)
                for t in toks[:-1]
            ]
        )
        self.play(
            FadeIn(toks),
            *cap(
                "Attention lets each move look back at every earlier token of its game."
            ),
            run_time=0.7,
        )
        self.play(
            last[0].animate.set_stroke(ORANGE, 4),
            LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.06),
            run_time=1.2,
        )
        self.play(
            *[Indicate(b[0], color=ORANGE, scale_factor=1.15) for b in blocks],
            run_time=0.8,
        )
        cap.hold(-0.3)

        legend = (
            VGroup(
                *[
                    VGroup(
                        Square(0.24, fill_color=c, fill_opacity=1, stroke_width=0),
                        sans(s, 24),
                    ).arrange(RIGHT, buff=0.12)
                    for c, s in (
                        (DENSE, "attention"),
                        (GREY, "dense MLP"),
                        (ORANGE, "mixture of experts"),
                    )
                ]
            )
            .arrange(RIGHT, buff=0.5)
            .move_to([0, -2.35, 0])
        )
        d = V0["dense"][-1]
        self.play(
            FadeOut(toks),
            FadeOut(arcs),
            *[
                b[1].animate.set_fill(GREY if i in dense else ORANGE)
                for i, b in enumerate(blocks, 1)
            ],
            FadeIn(legend),
            *cap(
                f"In Allie 2.0, block {d} has an ordinary MLP; "
                f"blocks {d + 1} to {n} use a mixture of experts."
            ),
            run_time=0.9,
        )
        cap.hold(-0.3)

        skips = VGroup(
            *[
                ArcBetweenPoints(
                    [xs[a - 1], yb - 0.5, 0], [xs[b - 1], yb - 0.5, 0], angle=PI * 0.36
                ).set_stroke(BLUE, 3.5)
                for a, b in D["skips"]
            ]
        )
        sk = ", ".join(f"{a}→{b}" for a, b in D["skips"])
        self.play(
            FadeOut(legend),
            LaggedStart(*[Create(s) for s in skips], lag_ratio=0.3),
            *cap(f"Shortcuts carry early blocks' outputs to later ones ({sk})."),
            run_time=1.2,
        )
        cap.hold(-0.3)

        pick = blocks[11]
        panel = RoundedRectangle(
            corner_radius=0.15,
            width=13.8,
            height=5.75,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=OFF_EDGE,
            stroke_width=2,
        ).move_to([0, 0.12, 0])
        self.play(
            Indicate(pick[1], color=ORANGE, scale_factor=1.3),
            *cap("Let's open one of the expert blocks."),
            run_time=0.8,
        )
        rest = [
            k,
            stream,
            skips,
            pick[0],
            pick[2],
            *[b for b in blocks if b is not pick],
        ]
        self.play(*[FadeOut(m) for m in rest], run_time=0.6)
        self.play(ReplacementTransform(pick[1], panel), *cap.clear(), run_time=0.9)


class Experts(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("4 · inside an expert block")
        panel = RoundedRectangle(
            corner_radius=0.15,
            width=13.8,
            height=5.75,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=OFF_EDGE,
            stroke_width=2,
        ).move_to([0, 0.12, 0])
        self.add(panel)
        Y = 0.1
        attn = box(1.3, 0.8, "attention", DENSE, px=24, at=[-5.75, Y, 0])
        router = Circle(radius=0.13, fill_color=INK, fill_opacity=1, stroke_width=0)
        router.move_to([-4.25, Y, 0])
        rlab = sans("router", 24).next_to(router, DOWN, buff=0.18)
        g = grid().move_to([-1.35, -0.42, 0])
        sh = shared_bar(g)
        glab = sans(f"{E} experts", 24).next_to(g, LEFT, buff=0.3).align_to(g, DOWN)
        merge = plus(0.22, [1.55, Y, 0])
        wires = VGroup(
            Line([-6.85, Y, 0], attn.get_left()),
            Line(attn.get_right(), router.get_center()),
            Line(router.get_center(), [g.get_left()[0], Y, 0]),
            Line([-3.75, Y, 0], [-3.75, sh.get_y(), 0]),
            Line([-3.75, sh.get_y(), 0], sh.get_left()),
            Line(sh.get_right(), [merge.get_x(), sh.get_y(), 0]),
            Line([merge.get_x(), sh.get_y(), 0], merge.get_top()),
            Line([g.get_right()[0], Y, 0], merge.get_left()),
            Line(merge.get_right(), [6.6, Y, 0]),
        ).set_stroke(RULE, 3)
        nxt = sans("to the next block", 22, MUTED).move_to([5.3, Y - 0.35, 0])
        self.play(
            FadeIn(k),
            FadeIn(VGroup(wires, attn, router, rlab, g, sh, glab, merge, nxt)),
            run_time=0.8,
        )

        v = vec(5, INK2, 4, 0.14).move_to([-6.6, Y + 0.3, 0])
        self.play(
            FadeIn(v, shift=0.2 * RIGHT),
            *cap("A move's vector passes attention, then reaches the router."),
            run_time=0.6,
        )
        self.play(
            v.animate.move_to(attn),
            Indicate(attn[0], color=INK, scale_factor=1.06),
            run_time=0.7,
        )
        self.play(v.animate.move_to([-4.25, Y + 0.32, 0]), run_time=0.5)
        cap.hold(-0.6)

        s, top, gates = route(11)
        self.play(
            *cap(f"The router scores all {E} experts for this move, each on its own."),
            LaggedStart(
                *[
                    c.animate.set_fill(shade(s[i])).set_stroke(shade(s[i]))
                    for i, c in enumerate(g)
                ],
                lag_ratio=0.004,
            ),
            run_time=1.3,
        )
        cap.hold(-0.4)
        on = set(top.tolist())
        self.play(
            *cap(f"The top {K} run, plus a shared expert that every move uses."),
            *light(g, on),
            sh[0].animate.set_fill(BLUE),
            sh[1].animate.set_color(BG),
            run_time=0.8,
        )
        cap.hold(-0.4)

        # the 16 gates as one bar 4 units long
        unit, bx, by = 1.0, 2.35, -1.45
        segs = VGroup()
        for j, gt in enumerate(gates):
            segs.add(
                Rectangle(
                    width=gt * unit,
                    height=0.4,
                    fill_color=ORANGE if j % 2 == 0 else PALE,
                    fill_opacity=1,
                    stroke_color=PANEL,
                    stroke_width=1.5,
                ).move_to([bx + unit * (gates[:j].sum() + gt / 2), by, 0])
            )
        ticks = VGroup(
            *[
                VGroup(
                    Line([bx + unit * t, by - 0.24, 0], [bx + unit * t, by - 0.32, 0]),
                ).set_stroke(RULE, 2)
                for t in range(D["gate_sum"] + 1)
            ],
            *[
                sans(str(t), 20, MUTED).move_to([bx + unit * t, by - 0.48, 0])
                for t in range(D["gate_sum"] + 1)
            ],
        )
        glab2 = sans(f"{K} gates add up to {D['gate_sum']}", 24, INK).next_to(
            segs, UP, buff=0.15
        )
        ins = [Dot(router.get_center(), 0.06, color=INK2) for _ in range(K)]
        sin = Dot([-3.75, Y, 0], 0.06, color=INK2)
        self.add(*ins, sin)
        self.play(
            *[d.animate.move_to(g[i]) for d, i in zip(ins, top)],
            sin.animate.move_to(sh.get_left()),
            FadeOut(v),
            run_time=0.7,
        )
        self.remove(*ins, sin)
        outs = [
            Dot(g[i].get_center(), 0.03 + 0.06 * gt, color=ORANGE)
            for i, gt in zip(top, gates)
        ]
        sout = Dot(sh.get_right(), 0.1, color=BLUE)
        self.add(*outs, sout)
        self.play(
            *cap(
                f"Their outputs are weighted by {K} gates that add up to {D['gate_sum']}, then summed."
            ),
            *[d.animate.move_to(merge) for d in outs],
            sout.animate.move_to(merge),
            LaggedStart(*[GrowFromEdge(r, LEFT) for r in segs], lag_ratio=0.08),
            FadeIn(ticks),
            FadeIn(glab2),
            run_time=1.3,
        )
        self.remove(*outs, sout)
        out = vec(5, ORANGE, 5, 0.14).move_to(merge)
        self.play(
            FadeIn(out, scale=0.5), merge[0].animate.set_stroke(ORANGE), run_time=0.4
        )
        self.play(out.animate.move_to([5.4, Y + 0.3, 0]), run_time=0.6)
        cap.hold(-0.3)
        idle = [c for i, c in enumerate(g) if i not in on]
        self.play(
            *cap(f"The other {E - K} experts do no work for this move."),
            *[c.animate.set_opacity(0.3) for c in idle],
            run_time=0.7,
        )
        cap.hold(-0.2)

        self.play(
            FadeOut(VGroup(out, segs, ticks, glab2)),
            merge[0].animate.set_stroke(INK2),
            *[c.animate.set_opacity(1).set_fill(OFF).set_stroke(OFF_EDGE) for c in g],
            *cap(f"Each move picks its own {K}, so the experts can specialise."),
            run_time=0.6,
        )
        for seed in (21, 37, 52):
            _, top2, _ = route(seed)
            v2 = vec(5, INK2, seed, 0.14).move_to([-6.6, Y + 0.3, 0])
            self.play(FadeIn(v2, shift=0.2 * RIGHT), run_time=0.25)
            self.play(v2.animate.move_to([-4.25, Y + 0.32, 0]), run_time=0.45)
            self.play(*light(g, set(top2.tolist())), FadeOut(v2), run_time=0.55)
            self.wait(0.25)
        cap.hold(-0.5)
        fade_all(self, cap)


class Heads(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("5 · three predictions")
        o = D["demo"]["outputs"]
        bd = board(o["fen"], 0.31).move_to([-5.15, 0.5, 0])
        blab = (
            VGroup(
                sans("Black to move", 24, INK),
                sans(f"{G['tc']} · White {G['white']} · Black {G['black']}", 20, MUTED),
            )
            .arrange(DOWN, buff=0.08)
            .next_to(bd, DOWN, buff=0.2)
        )
        fv = vec(8, ORANGE, 9, 0.19, vertical=True).move_to([-2.75, 0.3, 0])
        flab = sans(f"after block {D['blocks']}", 20, MUTED).next_to(
            fv, DOWN, buff=0.15
        )
        a0 = arrow(bd.get_right(), fv.get_left(), buff=0.15)
        self.play(
            FadeIn(k),
            FadeIn(bd),
            FadeIn(blab),
            *cap("Three heads read each move's final vector."),
            run_time=0.8,
        )
        self.play(Create(a0), FadeIn(fv, shift=0.2 * RIGHT), FadeIn(flab), run_time=0.6)
        cap.hold(-0.5)

        hx, ys = -1.6, [2.75, 0.55, -1.45]
        titles = [
            bold(s, 30).move_to([hx, y, 0], aligned_edge=LEFT)
            for s, y in zip(("Next move", "Think time", "Result"), ys)
        ]
        arrows = VGroup(
            *[arrow(fv.get_right(), t.get_left(), buff=0.15, width=2.5) for t in titles]
        )

        ms = o["moves"][:4]
        rows = hbars(
            [
                (san, p, ORANGE if j == 0 else PALE, INK)
                for j, (san, _, p) in enumerate(ms)
            ],
            0.3,
            ys[0] - 0.55,
            4.6 / ms[0][2],
            step=0.4,
            h=0.28,
            px=24,
        )
        self.play(
            Create(arrows[0]),
            FadeIn(titles[0]),
            *[FadeIn(r[0]) for r in rows],
            LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.15),
            *[FadeIn(r[2]) for r in rows],
            *cap(
                f"The next move: a probability for each of the {D['move_tokens']:,} move tokens."
            ),
            run_time=1.1,
        )
        cap.hold()

        pt, cen = np.array(o["time"]), np.array(D["demo"]["centres"])
        keep = cen <= 60
        pt, cen = pt[keep], cen[keep]
        ax0, ax1, base = -0.1, 5.8, ys[1] - 1.45
        tx = lambda t: ax0 + (ax1 - ax0) * np.log1p(t) / np.log1p(60)  # noqa: E731
        hist = VGroup(
            *[
                Rectangle(
                    width=(ax1 - ax0) / len(cen) * 0.85,
                    height=max(1.05 * p / pt.max(), 0.01),
                    fill_color=BLUE,
                    fill_opacity=0.9,
                    stroke_width=0,
                ).move_to([tx(t), base, 0], aligned_edge=DOWN)
                for p, t in zip(pt, cen)
            ]
        )
        axis = Line(
            [ax0 - 0.1, base, 0],
            [ax1 + 0.1, base, 0],
            stroke_color=RULE,
            stroke_width=2,
        )
        ticks = VGroup(
            *[
                sans(f"{t} s", 20, MUTED).move_to([tx(t), base - 0.22, 0])
                for t in (1, 10, 60)
            ]
        )
        mean = DashedLine(
            [tx(o["think"]), base, 0],
            [tx(o["think"]), base + 1.2, 0],
            color=ORANGE,
            stroke_width=2.5,
        )
        mlab = (
            sans(f"expected {o['think']:.1f} s", 22, ORANGE)
            .next_to(mean, RIGHT, buff=0.12)
            .align_to(mean, UP)
        )
        self.play(
            Create(arrows[1]),
            FadeIn(titles[1]),
            Create(axis),
            FadeIn(ticks),
            LaggedStart(*[GrowFromEdge(r, DOWN) for r in hist], lag_ratio=0.01),
            *cap(
                f"How long the player will think, in {D['heads']['time_bins']} time bins."
            ),
            run_time=1.1,
        )
        self.play(Create(mean), FadeIn(mlab), run_time=0.5)
        cap.hold()

        w, d, l_ = o["wdl"]
        bw, by = 5.9, ys[2] - 0.6
        segs, x = VGroup(), ax0
        for p, c in ((w, ORANGE), (d, GREY), (l_, BLUE)):
            segs.add(
                Rectangle(
                    width=bw * p,
                    height=0.44,
                    fill_color=c,
                    fill_opacity=1,
                    stroke_color=BG,
                    stroke_width=2,
                ).move_to([x, by, 0], aligned_edge=LEFT)
            )
            x += bw * p
        labs = VGroup(
            sans(f"win {pct(w)}", 22, ORANGE)
            .next_to(segs[0], DOWN, buff=0.12)
            .align_to(segs[0], LEFT),
            sans(f"draw {pct(d)}", 22, INK2).next_to(segs[1], DOWN, buff=0.12),
            sans(f"loss {pct(l_)}", 22, BLUE)
            .next_to(segs[2], DOWN, buff=0.12)
            .align_to(segs[2], RIGHT),
        )
        self.play(
            Create(arrows[2]),
            FadeIn(titles[2]),
            LaggedStart(*[GrowFromEdge(s_, LEFT) for s_ in segs], lag_ratio=0.3),
            FadeIn(labs),
            *cap("And how the game will end, for the player to move."),
            run_time=1.1,
        )
        cap.hold()
        aux = D["heads"]["aux_weight"]
        self.play(
            *cap(
                f"The move is the main training target; think time and result each weigh {aux}."
            ),
            run_time=0.6,
        )
        cap.hold()
        fade_all(self, cap)


class Human(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("6 · what human-like means")
        h = D["demo"]["human"]
        bd = board(h["fen"], 0.5).move_to([-3.9, 0.0, 0])
        line = mono(h["line"], 26, INK2).next_to(bd, UP, buff=0.2)
        self.play(
            FadeIn(k),
            FadeIn(bd),
            FadeIn(line),
            *cap(
                "Allie does not hunt for the best move. It predicts what a player would play."
            ),
            run_time=1.0,
        )
        cap.hold(-0.3)
        trap = arrow(at(bd, h["trap"][0]), at(bd, h["trap"][1]), BLUE, 0.1, 9, 0.26)
        threat = arrow(
            at(bd, h["threat"][0]), at(bd, h["threat"][1]), ORANGE, 0.1, 9, 0.26
        )
        self.play(
            Create(trap),
            *cap(
                f"Here Black can play {h['focus']}, which loses at once to {h['punish']}."
            ),
            run_time=0.8,
        )
        self.play(Create(threat), run_time=0.6)
        cap.hold(-0.4)

        ratings, ps = h["ratings"], h["p"]

        def bars(i):
            return hbars(
                [
                    (
                        s,
                        ps[i][s],
                        ORANGE if s == h["focus"] else BLUE_PALE,
                        ORANGE if s == h["focus"] else INK,
                    )
                    for s in h["show"]
                ],
                1.6,
                0.9,
                5.0,
                step=0.68,
                h=0.42,
                px=28,
                name_px=30,
            )

        rlab = sans("both players rated", 28).move_to([3.2, 2.2, 0], aligned_edge=RIGHT)

        def rnum(i):
            return bold(str(ratings[i]), 52, INK).next_to(rlab, RIGHT, buff=0.25)

        cur, num = bars(0), rnum(0)
        f0 = pct(ps[0][h["focus"]])
        self.play(
            FadeIn(rlab),
            FadeIn(num),
            FadeIn(cur),
            *cap(f"At {ratings[0]}, Allie expects {h['focus']} {f0} of the time."),
            run_time=0.9,
        )
        cap.hold(-0.3)
        self.play(*cap("Raise the rating, and the blunder fades."), run_time=0.5)
        for i in range(1, len(ratings)):
            anims = [Transform(cur, bars(i)), Transform(num, rnum(i))]
            if i == len(ratings) - 1:
                fl = pct(ps[i][h["focus"]])
                anims += cap(
                    f"At {ratings[i]}, only {fl}: strong players rarely fall for it."
                )
            self.play(*anims, run_time=0.8)
            self.wait(0.45)
        cap.hold(-0.2)

        self.play(
            *[FadeOut(m) for m in (bd, line, threat, trap, cur, rlab, num)],
            *cap.clear(),
            run_time=0.6,
        )
        items = [
            (True, "the move a person played"),
            (True, "how long they thought"),
            (True, "how their game ended"),
            (False, "an engine's evaluation"),
        ]
        rows = VGroup(
            *[
                VGroup(check(ok, 0.24), sans(s, 38, INK if ok else INK2)).arrange(
                    RIGHT, buff=0.4
                )
                for ok, s in items
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.36)
        head = bold("What Allie learns to predict", 42)
        VGroup(head, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(
            [0, 0.3, 0]
        )
        self.play(
            FadeIn(head),
            LaggedStart(
                *[FadeIn(r, shift=0.15 * RIGHT) for r in rows[:3]], lag_ratio=0.25
            ),
            *cap(
                "Every training target is a human decision: the move, the time taken, the result."
            ),
            run_time=1.2,
        )
        cap.hold(-0.3)
        eng = f"{D['human']['engine_share']:.0%}"
        self.play(
            FadeIn(rows[3], shift=0.15 * RIGHT),
            *cap(
                f"No engine evaluation is ever a target; engine games are under {eng} of the data."
            ),
            run_time=0.7,
        )
        cap.hold()
        self.play(
            *cap(
                "So Allie at rating R plays like a typical R-rated human, mistakes included."
            ),
            run_time=0.6,
        )
        cap.hold(0.3)
        fade_all(self, cap)


class Small(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("7 · why many small experts")
        mlp, ex, shw = V0["mlp"], V0["expert"], V0["shared"]
        W, x0, y = 11.0, -5.5, 2.0
        dense = Rectangle(
            width=W, height=0.6, fill_color=GREY, fill_opacity=1, stroke_width=0
        ).move_to([0, y, 0])
        dlab = (
            sans(f"one dense MLP: {mlp:,} hidden units", 28, INK)
            .next_to(dense, UP, buff=0.2)
            .align_to(dense, LEFT)
        )
        self.play(
            FadeIn(k),
            GrowFromEdge(dense, LEFT),
            FadeIn(dlab),
            *cap(f"Why {E} small experts? Start from the cost of one dense MLP."),
            run_time=1.0,
        )
        cap.hold(-0.5)
        u = W / mlp
        xs = x0 + u * np.r_[0, np.cumsum([ex] * K + [shw])]
        pieces = VGroup(
            *[
                Rectangle(
                    width=b - a - 0.05,
                    height=0.6,
                    fill_color=ORANGE if j < K else BLUE,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([(a + b) / 2, y, 0])
                for j, (a, b) in enumerate(zip(xs, xs[1:]))
            ]
        )
        lab1 = sans(f"{K} experts × {ex}", 26, ORANGE).next_to(
            VGroup(*pieces[:K]), DOWN, buff=0.15
        )
        lab2 = sans(f"shared {shw:,}", 26, BLUE).next_to(pieces[K], DOWN, buff=0.15)
        self.play(
            ReplacementTransform(dense, pieces),
            FadeIn(lab1),
            FadeIn(lab2),
            *cap(
                f"A move uses {K} experts of {ex} plus a shared one of {shw:,}: the same {mlp:,}."
            ),
            run_time=1.0,
        )
        cap.hold()

        cw = ex * u
        store = VGroup(
            *[
                Rectangle(
                    width=cw - 0.05,
                    height=0.15,
                    fill_color=OFF,
                    fill_opacity=1,
                    stroke_color=OFF_EDGE,
                    stroke_width=1,
                )
                for _ in range(E)
            ]
        ).arrange_in_grid(SIDE, SIDE, buff=(0.05, 0.06))
        store.move_to([x0 + store.width / 2, -1.0, 0])
        _, top, _ = route(8)
        times = E * ex // mlp
        slab = sans(f"stored: {E} experts,\n{times}× the dense MLP", 28, INK).next_to(
            store, RIGHT, buff=0.45
        )
        self.play(
            FadeIn(store),
            FadeIn(slab),
            *cap(
                f"But each layer stores {E} experts, {times}× the weights, and each move picks {K}."
            ),
            run_time=0.9,
        )
        picked = VGroup(
            *[store[i].copy().set_fill(ORANGE).set_stroke(ORANGE) for i in top]
        )
        self.play(
            *[TransformFromCopy(pieces[j], p) for j, p in enumerate(picked)],
            run_time=0.9,
        )
        cap.hold()

        self.play(
            FadeOut(VGroup(pieces, lab1, lab2, dlab, store, picked, slab)), run_time=0.5
        )
        wy = D["why"]
        rows = [
            (f"Allie: {E} small experts", 1.0, 1.0, ORANGE),
            (f"{wy['fine_vs']} experts, twice the size", *wy["fine"], PALE),
            ("dense transformer", *wy["dense"], GREY),
        ]
        bx, scale = -0.9, 2.5
        title = bold("training compute needed for the same loss", 30).move_to(
            [0, 2.5, 0]
        )
        axis = Line(
            [bx, -1.55, 0],
            [bx + scale * 2.8, -1.55, 0],
            stroke_color=RULE,
            stroke_width=2,
        )
        tks = VGroup(
            *[
                VGroup(
                    Line(
                        [bx + scale * t, -1.55, 0], [bx + scale * t, -1.65, 0]
                    ).set_stroke(RULE, 2),
                    sans(f"{t}×", 22, MUTED).move_to([bx + scale * t, -1.88, 0]),
                )
                for t in (1, 2)
            ]
        )
        bars = VGroup()
        for j, (name, lo, hi, col) in enumerate(rows):
            yy = 1.4 - 1.0 * j
            bar = Rectangle(
                width=scale * (lo + hi) / 2,
                height=0.55,
                fill_color=col,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([bx, yy, 0], aligned_edge=LEFT)
            rng = f"{lo:.2f}–{hi:.2f}×" if hi > lo else "1×"
            g = VGroup(
                sans(name, 26, INK).move_to([bx - 0.3, yy, 0], aligned_edge=RIGHT),
                bar,
                sans(rng, 24, INK2).move_to(
                    [bx + scale * hi + 0.2, yy, 0], aligned_edge=LEFT
                ),
            )
            if hi > lo:
                g.add(
                    Line([bx + scale * lo, yy, 0], [bx + scale * hi, yy, 0]).set_stroke(
                        INK, 3
                    )
                )
            bars.add(g)
        self.play(
            FadeIn(title),
            Create(axis),
            FadeIn(tks),
            FadeIn(bars[0]),
            *cap("How much training compute does each design need to match Allie?"),
            run_time=1.0,
        )
        cap.hold(-0.4)
        self.play(
            FadeIn(bars[1][0]),
            GrowFromEdge(bars[1][1], LEFT),
            FadeIn(bars[1][2:]),
            *cap(
                f"Experts twice the size, half as many: {wy['fine'][0]:.2f}–{wy['fine'][1]:.2f}× the compute."
            ),
            run_time=0.9,
        )
        cap.hold(-0.3)
        self.play(
            FadeIn(bars[2][0]),
            GrowFromEdge(bars[2][1], LEFT),
            FadeIn(bars[2][2:]),
            *cap(
                f"A dense transformer: {wy['dense'][0]:.2f}–{wy['dense'][1]:.2f}× the compute."
            ),
            run_time=0.9,
        )
        cap.hold(0.3)
        fade_all(self, cap)


class Router(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("8 · keeping the router healthy")
        steps = (
            VGroup(
                *[
                    sans(s, 26, MUTED)
                    for s in ("1 · collapse", "2 · overflow", "3 · built in")
                ]
            )
            .arrange(RIGHT, buff=0.9, aligned_edge=UP)
            .move_to([2.4, 3.5, 0])
        )
        g = grid(0.19, 0.045).move_to([-2.6, 0.1, 0])
        rng = np.random.default_rng(5)
        load = rng.uniform(0.25, 0.75, E)
        busy = [interpolate_color(ManimColor(OFF), ManimColor(ORANGE), x) for x in load]
        for c, col in zip(g, busy):
            c.set_fill(col).set_stroke(OFF_EDGE, 1)
        self.play(
            FadeIn(k),
            FadeIn(steps),
            FadeIn(g),
            *cap(
                "Each expert should get a fair share of moves. Routers can break that."
            ),
            run_time=0.9,
        )
        cap.hold(-0.4)

        self.play(steps[0].animate.set_color(ORANGE), run_time=0.3)
        n = ValueTracker(0)
        cnt = always_redraw(
            lambda: bold(f"{int(round(n.get_value()))}", 96, ORANGE).move_to(
                [3.4, 0.55, 0]
            )
        )
        clab = sans(f"of {E} experts starved", 30).move_to([3.4, -0.45, 0])
        self.add(cnt)
        self.play(
            FadeIn(clab),
            *cap(
                "In the first attempt, the first router's input became nearly constant."
            ),
            run_time=0.7,
        )
        cap.hold(-0.6)
        dead = rng.permutation(E)[: R["starved"]]
        self.play(
            n.animate.set_value(R["starved"]),
            LaggedStart(
                *[g[i].animate.set_fill(BG).set_stroke(RULE, 1) for i in dead],
                lag_ratio=0.01,
            ),
            *cap(
                f"Its scores froze, and {R['starved']} of {E} experts got no moves at all."
            ),
            run_time=2.2,
        )
        cap.hold(-0.3)
        lo, hi = R["starved_after"]
        self.play(
            n.animate.set_value(lo),
            LaggedStart(
                *[g[i].animate.set_fill(busy[i]).set_stroke(OFF_EDGE, 1) for i in dead],
                lag_ratio=0.01,
            ),
            *cap(
                "Fix: centre the router's input with a running mean, and step its optimizer every step."
            ),
            run_time=1.6,
        )
        after = bold(f"{lo}–{hi}", 96, BLUE).move_to([3.4, 0.55, 0])
        self.remove(cnt)
        self.add(after)
        self.play(Indicate(after, color=BLUE), run_time=0.6)
        cap.hold()

        self.play(
            FadeOut(VGroup(g, after, clab)),
            steps[0].animate.set_color(MUTED),
            steps[1].animate.set_color(ORANGE),
            run_time=0.5,
        )
        f1 = text(f"gate = {D['gate_sum']} · s / Σs", 52, INK, markup=True).move_to(
            [0, 1.3, 0]
        )
        tiny = text(f"Σs = {sci(R['overflow_sum'])}", 44, ORANGE, markup=True).move_to(
            [0, 0.0, 0]
        )
        boom = sans("the backward pass overflows", 32, ORANGE).next_to(
            tiny, DOWN, buff=0.35
        )
        self.play(
            FadeIn(f1),
            *cap(f"Each gate divides a score s by the sum of the {K} chosen scores."),
            run_time=0.7,
        )
        cap.hold(-0.4)
        self.play(
            FadeIn(tiny, shift=0.2 * DOWN),
            *cap(
                f"At step {R['overflow_step']:,}, one move's sum was {sci(R['overflow_sum'])}."
            ),
            run_time=0.8,
        )
        self.play(FadeIn(boom), Indicate(tiny, color=ORANGE), run_time=0.6)
        cap.hold(-0.3)
        f2 = text(
            f"gate = {D['gate_sum']} · s / max(Σs, {sci(R['floor'])})",
            52,
            INK,
            markup=True,
        ).move_to(f1)
        share = f"{R['floor_share']:.1%}"
        self.play(
            TransformMatchingShapes(f1, f2),
            FadeOut(boom),
            FadeOut(tiny),
            *cap(
                f"Fix: floor the sum at {sci(R['floor'])}. Only {share} of that layer's tokens change."
            ),
            run_time=1.0,
        )
        cap.hold()

        f3 = text(
            f"gate = {D['gate_sum']} · softmax(log s)", 52, INK, markup=True
        ).move_to(f1)
        self.play(
            steps[1].animate.set_color(MUTED),
            steps[2].animate.set_color(ORANGE),
            TransformMatchingShapes(f2, f3),
            *cap(
                "Allie 2.1 computes the same gates in log space, with no division at all."
            ),
            run_time=1.0,
        )
        cap.hold(-0.4)
        xs = np.linspace(-4.6, 4.6, D["blocks"])
        d21 = V1["dense"]
        strip = VGroup(
            *[
                RoundedRectangle(
                    corner_radius=0.04,
                    width=0.3,
                    height=0.85,
                    fill_color=GREY if i in d21 else ORANGE,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([x, -1.0, 0])
                for i, x in enumerate(xs, 1)
            ]
        )
        slab = (
            sans(f"Allie 2.1: blocks {d21[0]}–{d21[-1]} dense", 26)
            .next_to(strip, DOWN, buff=0.22)
            .align_to(strip, LEFT)
        )
        self.play(
            LaggedStart(*[FadeIn(b, shift=0.1 * UP) for b in strip], lag_ratio=0.03),
            FadeIn(slab),
            *cap(
                f"And its first {len(d21)} blocks are dense, so the most fragile routers are gone."
            ),
            run_time=1.0,
        )
        cap.hold()
        fade_all(self, cap)


class Params(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("9 · capacity is total, cost is active")
        s = 1.95
        big = Square(
            s * V0["total"] ** 0.5,
            fill_color=OFF,
            fill_opacity=1,
            stroke_color=RULE,
            stroke_width=2,
        ).move_to([-3.6, 0.0, 0])
        act = Square(
            s * V0["active"] ** 0.5, fill_color=ORANGE, fill_opacity=1, stroke_width=0
        ).align_to(big, DOWN + LEFT)
        tl = (
            VGroup(bold(f"{V0['total']}B", 48, INK), sans("parameters stored", 26))
            .arrange(DOWN, buff=0.08)
            .move_to(big.get_center() + 0.6 * UP + 0.5 * RIGHT)
        )
        al = (
            VGroup(bold(f"{V0['active']}B", 34, BG), sans("used per move", 20, BG))
            .arrange(DOWN, buff=0.05)
            .move_to(act)
        )
        self.play(
            FadeIn(k),
            FadeIn(big),
            FadeIn(tl),
            *cap(f"Allie 2.0 stores {V0['total']} billion parameters."),
            run_time=0.8,
        )
        cap.hold(-0.3)
        self.play(
            GrowFromEdge(act, DOWN + LEFT),
            FadeIn(al),
            *cap(
                f"Each move uses only {V0['active']} billion, since it runs just {K} experts per block."
            ),
            run_time=0.9,
        )
        cap.hold()

        rows = [
            ("Original Allie", D["original"]["gflops"], GREY),
            ("Allie 2.0", V0["gflops"], ORANGE),
            (D["maia3"]["name"], D["maia3"]["gflops"], BLUE_PALE),
        ]
        bx, sc = 0.9, 5.2 / D["maia3"]["gflops"]
        ttl = bold("compute per move, GFLOPs", 30).move_to(
            [bx, 2.0, 0], aligned_edge=LEFT
        )
        bars = VGroup()
        for j, (name, gf, col) in enumerate(rows):
            yy = 0.95 - 1.05 * j
            r_ = Rectangle(
                width=sc * gf,
                height=0.46,
                fill_color=col,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([bx, yy, 0], aligned_edge=LEFT)
            bars.add(
                VGroup(
                    sans(name, 24).next_to(r_, UP, buff=0.1).align_to(r_, LEFT),
                    r_,
                    sans(f"{gf}", 26, INK).next_to(r_, RIGHT, buff=0.14),
                )
            )
        self.play(
            FadeIn(ttl),
            *[FadeIn(b[0]) for b in bars],
            *[GrowFromEdge(b[1], LEFT) for b in bars],
            *[FadeIn(b[2]) for b in bars],
            *cap(
                f"So a move costs {V0['gflops']} GFLOPs, against {D['maia3']['gflops']} for {D['maia3']['name']}."
            ),
            run_time=1.1,
        )
        cap.hold()
        ms = D["cpu_ms"]
        self.play(
            *cap(
                f"A move reads only {D['int8_gb']} GB of int8 weights: {ms[0]}–{ms[1]} ms on a server CPU."
            ),
            run_time=0.6,
        )
        cap.hold()
        fade_all(self, cap)


class Next(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=WPS)
        k = kicker("10 · Allie 2.1")
        s, base = 1.15, -1.9
        sq, x = [], -6.5
        for v in (V0, V1):
            side = s * v["total"] ** 0.5
            big = Square(
                side, fill_color=OFF, fill_opacity=1, stroke_color=RULE, stroke_width=2
            )
            big.move_to([x + side / 2, base + side / 2, 0])
            act = Square(
                s * v["active"] ** 0.5,
                fill_color=ORANGE,
                fill_opacity=1,
                stroke_width=0,
            )
            act.align_to(big, DOWN + LEFT)
            lab = VGroup(
                sans(f"{v['active']}B active", 24, ORANGE),
                sans(f"{v['total']}B total", 24, INK2),
            ).arrange(DOWN, buff=0.06)
            lab.next_to(big, DOWN, buff=0.15)
            sq.append(VGroup(big, act, lab))
            x += side + 0.5
        heads = [
            bold(n, 32).next_to(q[0], UP, buff=0.2)
            for n, q in zip(("Allie 2.0", "Allie 2.1"), sq)
        ]

        def blocks(v):
            d = v["dense"]
            return f"{d[0]}" if len(d) == 1 else f"{d[0]}–{d[-1]}"

        rows = [
            ("width", f"{V0['width']:,}", f"{V1['width']:,}"),
            ("dense blocks", blocks(V0), blocks(V1)),
            ("training tokens", f"{V0['tokens']}B", f"{V1['tokens']}B"),
            ("GFLOPs per move", f"{V0['gflops']}", f"≈{V1['gflops']}"),
        ]
        cx = [1.2, 4.55, 6.1]
        hdr = VGroup(
            sans("2.0", 26, MUTED).move_to([cx[1], 2.0, 0]),
            sans("2.1", 26, ORANGE).move_to([cx[2], 2.0, 0]),
        )
        tab = VGroup()
        for j, (name, a, b) in enumerate(rows):
            yy = 1.25 - 0.75 * j
            tab.add(
                VGroup(
                    sans(name, 28).move_to([cx[0], yy, 0], aligned_edge=LEFT),
                    mono(a, 30, INK2).move_to([cx[1], yy, 0]),
                    mono(b, 30, INK).move_to([cx[2], yy, 0]),
                    Line([cx[0], yy - 0.37, 0], [cx[2] + 0.8, yy - 0.37, 0]).set_stroke(
                        OFF_EDGE, 1.5
                    ),
                )
            )
        self.play(
            FadeIn(k),
            FadeIn(sq[0]),
            FadeIn(heads[0]),
            *cap("Allie 2.0 is released. Allie 2.1 widens the same design."),
            run_time=0.8,
        )
        cap.hold(-0.6)
        self.play(
            GrowFromEdge(sq[1][0], DOWN + LEFT),
            GrowFromEdge(sq[1][1], DOWN + LEFT),
            FadeIn(sq[1][2]),
            FadeIn(heads[1]),
            *cap(
                f"{V1['active']}B active of {V1['total']}B, trained on {V1['tokens']}B tokens instead of {V0['tokens']}B."
            ),
            run_time=1.1,
        )
        self.play(
            FadeIn(hdr),
            LaggedStart(*[FadeIn(r, shift=0.1 * RIGHT) for r in tab], lag_ratio=0.2),
            run_time=1.0,
        )
        cap.hold()
        self.play(
            *cap(
                f"Allie 2.1 has been {V1['status']}, with the router guards built in."
            ),
            run_time=0.6,
        )
        cap.hold()
        fade_all(self, cap)

        name = text("Allie", 92, INK, weight="SEMIBOLD")
        links = VGroup(
            mono("github.com/y0mingzhang/allie", 32, INK2),
            mono("huggingface.co/yimingzhang/allie-2.0", 32, INK2),
            mono("lichess.org/@/AllieTheChessBot", 32, ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        VGroup(name, links).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(
            [0, 0.3, 0]
        )
        self.play(
            FadeIn(name, shift=0.2 * UP),
            LaggedStart(*[FadeIn(m) for m in links], lag_ratio=0.2),
            *cap("Code, weights, and a bot to play on Lichess."),
            run_time=1.2,
        )
        cap.hold(1.2)
        fade_all(self, cap, 1.0)
