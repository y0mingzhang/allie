"""How Allie Searches: why the raw policy needs search, the calibration target, one output
rule and two searchers, the shipped per-cell results, the value-noise ceiling in strong
classical, and what plays on Lichess. Every number comes from DATA["search"]."""

import numpy as np
from manim import (
    AnimationGroup,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Group,
    GrowFromCenter,
    GrowFromEdge,
    LaggedStart,
    Line,
    ManimColor,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    ShowPassingFlash,
    Square,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    interpolate_color,
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

D = DATA["search"]
S = D["schematic"]
CELLS, FMTS, BINS = D["cells"], D["formats"], D["bins"]
PALE_BLUE, BLUE_TXT = "#9DB7CC", "#5F86A5"
VIDEO = dict(
    name="how-allie-searches",
    scenes=["Title", "Why", "Target", "Rule", "Trees", "Results", "Ceiling", "Live"],
    gif=(["Trees", 7.5], 15),
    poster=["Title", 5.0],
)


def num(x, fmt="{:+d}"):
    return fmt.format(x).replace("-", "−")


def mix(a, b, t):
    return interpolate_color(ManimColor(a), ManimColor(b), float(np.clip(t, 0, 1)))


def tilt(p, q, beta=0.0, t=1.0):
    w = np.asarray(p) ** (1 / t) * np.exp(beta * np.asarray(q))
    return w / w.sum()


class Plot:
    """Data to scene coordinates inside box = (left, right, bottom, top)."""

    def __init__(self, xr, yr, box, xlog=False):
        self.xr, self.yr, self.box, self.xlog = xr, yr, box, xlog

    def x(self, v):
        a, b = self.xr
        if self.xlog:
            v, a, b = np.log2(v), np.log2(a), np.log2(b)
        return self.box[0] + (v - a) / (b - a) * (self.box[1] - self.box[0])

    def y(self, v):
        a, b = self.yr
        return self.box[2] + (v - a) / (b - a) * (self.box[3] - self.box[2])

    def p(self, x, y):
        return np.array([self.x(x), self.y(y), 0])

    def frame(self, xticks, yticks, xfmt=str, yfmt=str, xlabel="", ylabel=""):
        x0, r, b, t = self.box
        g = VGroup()
        for v in yticks:
            g.add(
                Line(
                    [x0, self.y(v), 0],
                    [r, self.y(v), 0],
                    stroke_color=PANEL,
                    stroke_width=2,
                )
            )
            g.add(text(yfmt(v), 18, MUTED).next_to([x0, self.y(v), 0], LEFT, buff=0.15))
        g.add(Line([x0, b, 0], [r, b, 0], stroke_color=RULE, stroke_width=2))
        for v in xticks:
            g.add(
                Line(
                    [self.x(v), b, 0],
                    [self.x(v), b - 0.08, 0],
                    stroke_color=RULE,
                    stroke_width=2,
                )
            )
            g.add(
                text(xfmt(v), 18, MUTED).next_to(
                    [self.x(v), b - 0.08, 0], DOWN, buff=0.08
                )
            )
        if xlabel:
            g.add(text(xlabel, 20, INK2).move_to([(x0 + r) / 2, b - 0.62, 0]))
        if ylabel:
            g.add(
                text(ylabel, 20, INK2).move_to(
                    [x0 + 0.02, t + 0.32, 0], aligned_edge=LEFT
                )
            )
        return g

    def line(self, xs, ys, color, width=4, r=0.055):
        pts = [self.p(x, y) for x, y in zip(xs, ys)]
        path = VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(
            pts
        )
        dots = VGroup(*[Dot(q, radius=r, color=color) for q in pts])
        return VGroup(path, dots)


# 4 x 10 grid of time control x rating cells
CW, CH, GAP, X0, Y0 = 1.1, 0.66, 0.08, -4.72, 1.25


def cxy(r, c):
    return np.array([X0 + c * (CW + GAP), Y0 - r * (CH + GAP), 0])


def grid_labels():
    rows = VGroup(
        *[
            text(f, 24, INK2).move_to(
                cxy(r, 0) + LEFT * (CW / 2 + 0.16), aligned_edge=RIGHT
            )
            for r, f in enumerate(FMTS)
        ]
    )
    cols = VGroup(
        *[
            text(str(b), 20, MUTED).move_to(cxy(0, c) + UP * (CH / 2 + 0.24))
            for c, b in enumerate(BINS)
        ]
    )
    return VGroup(rows, cols)


def tile(r, c, s, fill, color=INK, sub=None, stroke=None):
    box = RoundedRectangle(
        corner_radius=0.06,
        width=CW,
        height=CH,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=stroke or fill,
        stroke_width=3 if stroke else 0,
    ).move_to(cxy(r, c))
    t = text(s, 22, color, font=MONO, weight="MEDIUM")
    if sub is None:
        return VGroup(box, t.move_to(box))
    sb = text(sub, 15, color, weight="MEDIUM")
    return VGroup(box, VGroup(t, sb).arrange(DOWN, buff=0.04).move_to(box))


def heat(e):
    """Fill and text colour for an Elo gap: orange too weak, blue too strong."""
    t = min(abs(e) / 500, 1)
    return mix(OFF, ORANGE if e < 0 else BLUE, 0.9 * t), OFF if t > 0.55 else INK


FEN = "r1bq1rk1/pp2bpp1/2np1n1p/4p3/2BPP3/2P2N2/PP1N1PPP/R1BQ1RK1"  # white to move: S["moves"]
GLYPH = dict(zip("KQRBNPkqrbnp", "♔♕♖♗♘♙♚♛♜♝♞♟"))


def board(size=2.4):
    sq = size / 8
    at = lambda i, j: [(j - 3.5) * sq, (3.5 - i) * sq, 0]  # noqa: E731
    g = VGroup(
        *[
            Square(
                sq,
                fill_color=OFF_EDGE if (i + j) % 2 else PANEL,
                fill_opacity=1,
                stroke_width=0,
            ).move_to(at(i, j))
            for i in range(8)
            for j in range(8)
        ]
    )
    for i, row in enumerate(FEN.split("/")):
        j = 0
        for ch in row:
            if ch.isdigit():
                j += int(ch)
                continue
            g.add(
                text(GLYPH[ch], 27, INK, font="DejaVu Sans", weight="NORMAL").move_to(
                    at(i, j)
                )
            )
            j += 1
    return g


def allie_box(label="Allie"):
    b = RoundedRectangle(
        corner_radius=0.12,
        width=1.9,
        height=1.0,
        fill_color=OFF,
        fill_opacity=1,
        stroke_color=ORANGE,
        stroke_width=3,
    )
    return VGroup(b, text(label, 32, INK, weight="SEMIBOLD").move_to(b))


class Bars(VGroup):
    """The schematic position's 8 moves: bars of a distribution over move labels."""

    def __init__(self, x0, dx, base, scale, width=0.44, q=False, human=False):
        super().__init__()
        self.xs = [x0 + i * dx for i in range(len(S["p"]))]
        self.y0, self.k, self.bw = base, scale, width
        self.bars = self.make(S["p"], RULE)
        self.labels = VGroup(
            *[
                text(m, 20, INK, font=MONO, weight="MEDIUM").move_to(
                    [x, base - 0.26, 0]
                )
                for x, m in zip(self.xs, S["moves"])
            ]
        )
        self.add(self.bars, self.labels)
        if q:
            self.q = VGroup(
                *[
                    text(num(v, "{:+.2f}"), 17, BLUE, font=MONO).move_to(
                        [x, base - 0.58, 0]
                    )
                    for x, v in zip(self.xs, S["q"])
                ]
            )
        if human:
            self.marker = VGroup(
                text("human's move", 18, INK, weight="SEMIBOLD"),
                Arrow(
                    UP * 0.3,
                    DOWN * 0.1,
                    buff=0,
                    stroke_width=3,
                    color=INK,
                    max_tip_length_to_length_ratio=0.35,
                ),
            ).arrange(DOWN, buff=0.05)

    def make(self, p, color):
        return VGroup(
            *[
                Rectangle(
                    width=self.bw,
                    height=max(v * self.k, 0.01),
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([x, self.y0, 0], aligned_edge=DOWN)
                for x, v in zip(self.xs, p)
            ]
        )

    def place_marker(self, p):
        h = S["human"]
        return self.marker.move_to([self.xs[h], self.y0 + p[h] * self.k + 0.42, 0])


class Gauge(VGroup):
    """Cross-entropy of the human's move under a distribution, against the raw policy's."""

    def __init__(self, left, y, k=0.95, label="surprise at the human's move  (CE)"):
        super().__init__()
        self.x0, self.y0, self.k = left, y, k
        self.raw = -np.log(S["p"][S["human"]])
        self.title = text(label, 20, INK2).move_to(
            [left, y + 0.55, 0], aligned_edge=LEFT
        )
        x = left + k * self.raw
        self.tick = Line(
            [x, y - 0.35, 0], [x, y + 0.32, 0], stroke_color=INK, stroke_width=3
        )
        self.tlabel = text("raw policy", 18, INK).next_to(self.tick, DOWN, buff=0.08)
        self.add(self.title, self.tick, self.tlabel)

    def bar(self, ce):
        color = (
            RULE if abs(ce - self.raw) < 1e-6 else (BLUE if ce < self.raw else ORANGE)
        )
        return Rectangle(
            width=self.k * ce,
            height=0.36,
            fill_color=color,
            fill_opacity=1,
            stroke_width=0,
        ).move_to([self.x0, self.y0, 0], aligned_edge=LEFT)


def node(x, y, v, r, hi):
    return Circle(
        radius=r,
        fill_color=mix("#E2DDD3", hi, (v + 0.5) / 1.0),
        fill_opacity=1,
        stroke_color=RULE,
        stroke_width=1.5,
    ).move_to([x, y, 0])


def hollow(x, y, r):
    return Circle(
        radius=r, fill_color=BG, fill_opacity=1, stroke_color=OFF_EDGE, stroke_width=1.5
    ).move_to([x, y, 0])


def edge(a, b, dashed=False):
    cls = DashedLine if dashed else Line
    return cls(
        a.get_center(),
        b.get_center(),
        buff=a.radius,
        stroke_color=OFF_EDGE if dashed else RULE,
        stroke_width=2,
    )


def unstack(labels, gap=0.08):
    """Nudges right-end line labels apart vertically so none overlap."""
    order = sorted(labels, key=lambda m: -m.get_y())
    for a, b in zip(order, order[1:]):
        lo = a.get_bottom()[1] - gap
        if b.get_top()[1] > lo:
            b.shift((b.get_top()[1] - lo) * DOWN)
    return labels


def swap(old, new):
    """Crossfade one mobject into another in place (no glyph morphing)."""
    return AnimationGroup(FadeOut(old), FadeIn(new))


def schematic():
    """Top-right tag for beats drawn with illustrative numbers."""
    return text("schematic", 20, MUTED).to_corner(UP + RIGHT, buff=0.45)


def wipe(scene, cap=None, keep=()):
    """Fade out everything on screen but `keep`, the caption included."""
    held = [*keep, cap.cur] if cap else list(keep)
    mobs = [m for m in scene.mobjects if all(m is not h for h in held)]
    return [FadeOut(Group(*mobs)), *(cap.clear() if cap else [])]


class Title(Scene):
    def construct(self):
        paper(self)
        title = VGroup(
            text("How Allie", 76, INK, weight="SEMIBOLD"),
            text("Searches", 76, ORANGE, weight="SEMIBOLD"),
        )
        title.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(
            [-6.1, 0.9, 0], aligned_edge=LEFT
        )
        sub = VGroup(
            *[
                text(s, 26, INK2)
                for s in (
                    "Search that plays close to the",
                    "strength of the humans Allie imitates,",
                    "and still predicts their moves",
                )
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        sub.next_to(title, DOWN, aligned_edge=LEFT, buff=0.45)
        tag = text("Allie 2.0 · github.com/y0mingzhang/allie", 18, MUTED).move_to(
            [-6.1, -3.2, 0], aligned_edge=LEFT
        )

        cx, top = 3.6, 2.5
        p, q = np.array(S["p"]), np.array(S["q"])
        xs = cx + np.linspace(-2.6, 2.6, len(p))
        root = Circle(
            radius=0.16, fill_color=INK, fill_opacity=1, stroke_width=0
        ).move_to([cx, top, 0])
        kids = [node(x, 1.35, v, 0.13, ORANGE) for x, v in zip(xs, q)]
        e1 = [edge(root, k) for k in kids]
        rng = np.random.default_rng(1)
        deep, e2 = [], []
        for i in (0, 2, 3, 5):
            for dx in (-0.16, 0.16):
                c = node(xs[i] + dx, 0.35, q[i] + rng.normal(0, 0.2), 0.09, ORANGE)
                deep.append(c)
                e2.append(edge(kids[i], c))
        base, scale = -2.5, 4.2
        pi = tilt(p, q, S["beta"])
        prior = VGroup(
            *[
                Rectangle(
                    width=0.36,
                    height=v * scale,
                    fill_opacity=0,
                    stroke_color=RULE,
                    stroke_width=2.5,
                ).move_to([x, base, 0], aligned_edge=DOWN)
                for x, v in zip(xs, p)
            ]
        ).set_z_index(1)
        post = VGroup(
            *[
                Rectangle(
                    width=0.36,
                    height=v * scale,
                    fill_color=ORANGE,
                    fill_opacity=1,
                    stroke_width=0,
                ).move_to([x, base, 0], aligned_edge=DOWN)
                for x, v in zip(xs, pi)
            ]
        )
        axis = Line(
            [xs[0] - 0.35, base, 0],
            [xs[-1] + 0.35, base, 0],
            stroke_color=RULE,
            stroke_width=2,
        )
        self.play(FadeIn(title, shift=0.2 * UP), run_time=0.9)
        self.play(
            FadeIn(sub, shift=0.1 * UP),
            GrowFromCenter(root),
            LaggedStart(*[GrowFromCenter(m) for m in (*e1, *kids)], lag_ratio=0.03),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in (*e2, *deep)], lag_ratio=0.03),
            Create(axis),
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in prior], lag_ratio=0.05),
            FadeIn(tag),
            run_time=1.1,
        )
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in post], lag_ratio=0.05),
            run_time=1.0,
        )
        self.wait(2.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
        self.wait(0.2)


class Why(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        k = kicker("Why search")
        b = board().move_to([-4.8, 0.1, 0])
        blabel = text("position · both ratings · clocks", 20, MUTED).next_to(
            b, DOWN, buff=0.22
        )
        allie = allie_box().move_to([-1.35, 0.1, 0])
        bars = Bars(1.25, 0.72, -1.3, 7.0)
        a1 = Arrow(
            b.get_right(),
            allie.get_left(),
            buff=0.18,
            color=RULE,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.2,
        )
        a2 = Arrow(
            allie.get_right(),
            [bars.xs[0] - 0.4, 0.1, 0],
            buff=0.18,
            color=RULE,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.2,
        )
        self.play(
            FadeIn(k),
            FadeIn(b),
            FadeIn(blabel),
            *cap("Allie predicts the move a human of a given rating would play"),
        )
        self.play(GrowArrow(a1), FadeIn(allie), run_time=0.8)
        self.play(
            GrowArrow(a2),
            LaggedStart(*[GrowFromEdge(r, DOWN) for r in bars.bars], lag_ratio=0.06),
            FadeIn(bars.labels),
            run_time=1.2,
        )
        cap.hold()
        self.play(
            *cap(
                "Sample a move straight from that prediction (T = 1): that is the raw policy"
            )
        )
        self.play(bars.bars[0].animate.set_fill(ORANGE), run_time=0.5)
        cap.hold()
        self.play(FadeOut(VGroup(b, blabel, a1, a2, allie, bars)), run_time=0.7)

        P = Plot((800, 2600), (84, 96), (-5.3, 4.6, -1.95, 2.0))
        fr = P.frame(
            BINS,
            [84, 88, 92, 96],
            yfmt=lambda v: f"{v}%",
            xlabel="rating",
            ylabel="move accuracy, classical games",
        )
        acc = lambda key: [CELLS[f"classical/{v}"][key] for v in BINS]  # noqa: E731
        hum, raw = (
            P.line(BINS, acc("human_acc"), INK),
            P.line(BINS, acc("raw_acc"), DENSE),
        )
        hl = text("humans", 22, INK, weight="SEMIBOLD").next_to(
            hum[1][-1], RIGHT, buff=0.15
        )
        rl = text("Allie, raw", 22, DENSE, weight="SEMIBOLD").next_to(
            raw[1][-1], RIGHT, buff=0.15
        )
        self.play(
            FadeIn(fr),
            *cap(
                "Stockfish scores every move: humans get more accurate as their rating rises"
            ),
            run_time=0.8,
        )
        self.play(
            Create(hum[0]),
            LaggedStart(*[GrowFromCenter(d) for d in hum[1]], lag_ratio=0.08),
            FadeIn(hl),
            run_time=1.6,
        )
        cap.hold()
        self.play(
            *cap("Allie's raw policy keeps pace at low ratings, then falls behind")
        )
        self.play(
            Create(raw[0]),
            LaggedStart(*[GrowFromCenter(d) for d in raw[1]], lag_ratio=0.08),
            FadeIn(rl),
            run_time=1.6,
        )
        top, bot = hum[1][-1].get_center(), raw[1][-1].get_center()
        gap = VGroup(
            Line(top + 0.1 * DOWN, bot + 0.1 * UP, stroke_color=ORANGE, stroke_width=4),
            text(
                f"{acc('human_acc')[-1]:.1f}% vs {acc('raw_acc')[-1]:.1f}%",
                20,
                ORANGE,
                weight="SEMIBOLD",
            ).next_to((top + bot) / 2, LEFT, buff=0.18),
        )
        self.play(FadeIn(gap), run_time=0.6)
        cap.hold()

        E = Plot((800, 2600), (-650, 100), P.box)
        fr2 = E.frame(
            BINS,
            [0, -200, -400, -600],
            yfmt=lambda v: num(v) if v else "0",
            xlabel="rating",
            ylabel="Elo, Allie's raw policy vs humans",
        )
        shades = {"blitz": "#AEB6BD", "rapid": "#848E97", "classical": DENSE}
        lines = {
            f: E.line(
                BINS,
                [CELLS[f"{f}/{v}"]["raw_elo"] for v in BINS],
                shades[f],
                width=3.5,
                r=0.045,
            )
            for f in shades
        }
        names = VGroup(
            *[
                text(f, 20, shades[f], weight="SEMIBOLD").next_to(
                    lines[f][1][-1], RIGHT, buff=0.15
                )
                for f in shades
            ]
        )
        unstack(names)
        zero = Line(E.p(800, 0), E.p(2600, 0), stroke_color=INK, stroke_width=4)
        zl = text("humans", 20, INK, weight="SEMIBOLD").next_to(zero, RIGHT, buff=0.15)
        means = [-D["raw"][f]["mean"] for f in ("blitz", "rapid", "classical")]
        worst = min(
            -CELLS[f"{f}/2600"]["raw_elo"] for f in ("blitz", "rapid", "classical")
        )
        self.play(
            FadeOut(fr),
            FadeIn(fr2),
            FadeOut(hum),
            FadeOut(hl),
            FadeIn(zero),
            FadeIn(zl),
            ReplacementTransform(raw, lines["classical"]),
            swap(rl, names[2]),
            FadeOut(gap),
            *cap(
                f"In Elo: {min(means)} to {max(means)} weaker on average, and over {worst // 100 * 100} weaker at 2600"
            ),
            run_time=1.4,
        )
        self.play(
            Create(lines["blitz"]),
            Create(lines["rapid"]),
            FadeIn(names[0]),
            FadeIn(names[1]),
            run_time=1.2,
        )
        cap.hold(0.5)
        self.play(
            FadeOut(VGroup(fr2, zero, zl, *lines.values(), names)),
            *cap(
                "Strong players calculate before they move. Search adds the calculation"
            ),
            run_time=0.8,
        )
        allie = allie_box().move_to([-2.2, 0.4, 0])
        root = Dot([0.4, 0.4, 0], radius=0.11, color=INK)
        link = Line(
            allie.get_right(), root.get_center(), stroke_color=RULE, stroke_width=3
        )
        rng = np.random.default_rng(5)
        kids = [
            node(1.7, 0.4 + y, rng.uniform(-0.5, 0.5), 0.09, ORANGE)
            for y in np.linspace(-1.5, 1.5, 7)
        ]
        ek = [
            Line(
                root.get_center(),
                c.get_center(),
                buff=0.09,
                stroke_color=RULE,
                stroke_width=2,
            )
            for c in kids
        ]
        gk, eg = [], []
        for c in kids[1:6]:
            for dy in (-0.14, 0.14):
                g = node(3.0, c.get_y() + dy, rng.uniform(-0.5, 0.5), 0.07, ORANGE)
                gk.append(g)
                eg.append(
                    Line(
                        c.get_center(),
                        g.get_center(),
                        buff=0.07,
                        stroke_color=RULE,
                        stroke_width=2,
                    )
                )
        self.play(FadeIn(allie), Create(link), GrowFromCenter(root), run_time=0.6)
        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in (*ek, *kids)], lag_ratio=0.04),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in (*eg, *gk)], lag_ratio=0.03),
            run_time=0.9,
        )
        cap.hold()
        self.play(*wipe(self, cap), run_time=0.8)
        self.wait(0.2)


class Target(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        k = kicker("The calibration target")
        labels = grid_labels()

        def acc_tile(r, c):
            v = CELLS[f"{FMTS[r]}/{BINS[c]}"]["human_acc"]
            t = (v - 84) / 11
            return tile(
                r, c, f"{v:.1f}", mix(OFF, DENSE, 0.85 * t), OFF if t > 0.6 else INK
            )

        def blunder_tile(r, c):
            v = CELLS[f"{FMTS[r]}/{BINS[c]}"]["human_blunder"]
            t = v / 12
            return tile(
                r, c, f"{v:.1f}", mix(OFF, ORANGE, 0.9 * t), OFF if t > 0.6 else INK
            )

        idx = [(r, c) for r in range(len(FMTS)) for c in range(len(BINS))]
        accs = VGroup(*[acc_tile(r, c) for r, c in idx])
        blunders = VGroup(*[blunder_tile(r, c) for r, c in idx])
        unit = text("human move accuracy, %", 22, INK2, weight="SEMIBOLD").move_to(
            [X0 - CW / 2, -1.85, 0], aligned_edge=LEFT
        )
        unit2 = text(
            "human blunder rate, % of moves", 22, ORANGE, weight="SEMIBOLD"
        ).move_to(unit, aligned_edge=LEFT)
        npos = f"{D['positions']:,}"
        self.play(
            FadeIn(k),
            FadeIn(labels),
            *cap(
                f"{npos} Lichess positions from {D['month']}, in {D['n_cells']} cells: time control × rating"
            ),
        )
        self.play(
            LaggedStart(*[FadeIn(t, scale=0.9) for t in accs], lag_ratio=0.015),
            run_time=1.6,
        )
        cap.hold()
        self.play(
            FadeIn(unit),
            *cap(
                "Each cell sets a target: how accurately its humans move, scored by Stockfish"
            ),
        )
        cap.hold()
        self.play(
            LaggedStart(*[swap(a, b) for a, b in zip(accs, blunders)], lag_ratio=0.012),
            swap(unit, unit2),
            *cap(
                f"and how often they blunder: give up {D['blunder_drop']} or more points of win chance"
            ),
            run_time=1.2,
        )
        cap.hold()

        cell = CELLS["rapid/2400"]
        r, c = FMTS.index("rapid"), BINS.index(2400)
        focus = blunders[r * len(BINS) + c]
        self.play(
            FadeOut(VGroup(labels, unit2, *[t for t in blunders if t is not focus])),
            focus.animate.move_to([-5.2, 2.0, 0]),
            run_time=0.9,
        )
        head = text("rapid · 2400", 30, INK, weight="SEMIBOLD").next_to(
            focus, RIGHT, buff=0.3
        )
        cols = VGroup(
            text("humans", 24, INK, weight="SEMIBOLD"),
            text("Allie, raw policy", 24, DENSE, weight="SEMIBOLD"),
        )
        cols[0].move_to([-0.6, 1.0, 0])
        cols[1].move_to([3.6, 1.0, 0])
        rows = VGroup(text("move accuracy", 24, INK2), text("blunder rate", 24, INK2))
        rows[0].move_to([-5.7, 0.1, 0], aligned_edge=LEFT)
        rows[1].move_to([-5.7, -1.3, 0], aligned_edge=LEFT)
        big = lambda v, col: text(f"{v:.1f}%", 44, col, font=MONO, weight="MEDIUM")  # noqa: E731
        vals = VGroup(
            big(cell["human_acc"], INK).move_to([-0.6, 0.1, 0]),
            big(cell["raw_acc"], DENSE).move_to([3.6, 0.1, 0]),
            big(cell["human_blunder"], INK).move_to([-0.6, -1.15, 0]),
            big(cell["raw_blunder"], ORANGE).move_to([3.6, -1.15, 0]),
        )
        kb = 0.32
        bb = VGroup(
            Rectangle(
                width=kb * cell["human_blunder"],
                height=0.16,
                fill_color=INK,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([-0.6, -1.72, 0]),
            Rectangle(
                width=kb * cell["raw_blunder"],
                height=0.16,
                fill_color=ORANGE,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([3.6, -1.72, 0]),
        )
        ratio = cell["raw_blunder"] / cell["human_blunder"]
        self.play(
            FadeIn(head),
            FadeIn(cols),
            FadeIn(rows),
            *cap(
                f"Rapid 2400: the raw policy blunders {ratio:.1f}× as often as the humans it imitates"
            ),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(v, shift=0.1 * UP) for v in vals], lag_ratio=0.2),
            LaggedStart(*[GrowFromEdge(x, LEFT) for x in bb], lag_ratio=0.3),
            run_time=1.4,
        )
        cap.hold(0.6)
        self.play(FadeOut(VGroup(focus, head, cols, rows, vals, bb)), run_time=0.7)

        bars = Bars(-5.6, 0.75, -2.0, 4.0, width=0.5, human=True)
        p = np.array(S["p"])
        g = Gauge(1.2, -1.0)
        ce = lambda d: -np.log(d[S["human"]])  # noqa: E731
        gb = g.bar(ce(p))
        self.play(
            LaggedStart(*[GrowFromEdge(x, DOWN) for x in bars.bars], lag_ratio=0.05),
            FadeIn(bars.labels),
            FadeIn(bars.place_marker(p)),
            FadeIn(schematic()),
            *cap("Strength alone is easy. Allie must also keep predicting human moves"),
            run_time=1.0,
        )
        self.play(FadeIn(g), GrowFromEdge(gb, LEFT), run_time=0.8)
        cap.hold()

        def show(d, color, caption, rt=1.2):
            self.play(
                Transform(bars.bars, bars.make(d, color)),
                bars.marker.animate.move_to(bars.place_marker(d).get_center()),
                Transform(gb, g.bar(ce(d))),
                *cap(caption),
                run_time=rt,
            )
            cap.hold()

        show(
            tilt(p, S["q"], S["beta_engine"]),
            ORANGE,
            "Always play the engine's favorite: strong, but the human's move becomes a surprise",
        )
        show(
            tilt(p, S["q"], t=S["t_sharp"]),
            ORANGE,
            "Sharpen the prediction (temperature below 1): the human's move gets less likely too",
        )
        show(
            p,
            RULE,
            "The bar: in every searching cell, cross-entropy below the raw policy's, at T = 1",
        )
        self.play(*wipe(self, cap), run_time=0.8)
        self.wait(0.2)


class Rule(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        k = kicker("One output rule")
        bars = Bars(-5.6, 0.75, -2.0, 4.0, width=0.5, q=True, human=True)
        bars.labels.shift(0.08 * UP)
        p, q = np.array(S["p"]), np.array(S["q"])
        g = Gauge(1.2, -1.0)
        beta = ValueTracker(0)
        ce = lambda d: -np.log(d[S["human"]])  # noqa: E731
        cur = lambda: tilt(p, q, beta.get_value())  # noqa: E731
        gb = always_redraw(lambda: g.bar(ce(cur())))
        self.play(
            FadeIn(k),
            FadeIn(bars.bars),
            FadeIn(bars.labels),
            FadeIn(bars.place_marker(p)),
            FadeIn(g),
            FadeIn(gb),
            FadeIn(schematic()),
            run_time=0.6,
        )
        qhead = text("Q", 20, BLUE, font=MONO, weight="MEDIUM").next_to(
            bars.q[0], LEFT, buff=0.25
        )
        self.play(
            FadeIn(bars.q, lag_ratio=0.1),
            FadeIn(qhead),
            *cap(
                "Search gives each move a value Q: how well it turns out for the mover"
            ),
            run_time=1.2,
        )
        cap.hold()
        formula = text(
            f'π(a) ∝ p(a) · exp(<span foreground="{ORANGE}">β</span> · <span foreground="{BLUE}">Q(a)</span>)',
            40,
            INK,
            font=MONO,
            weight="MEDIUM",
            markup=True,
        ).move_to([0, 2.55, 0])
        readout = always_redraw(
            lambda: text(
                f"β = {beta.get_value():.1f}", 34, ORANGE, font=MONO, weight="MEDIUM"
            ).move_to([1.2, 0.3, 0], aligned_edge=LEFT)
        )
        self.play(
            FadeIn(formula, shift=0.15 * DOWN),
            *cap(
                "One rule turns any search into moves: tilt the prediction p by exp(βQ)"
            ),
            run_time=1.0,
        )
        cap.hold()
        bars.bars.add_updater(
            lambda m: m.become(
                bars.make(cur(), mix(RULE, ORANGE, beta.get_value() / 1.5))
            )
        )
        bars.marker.add_updater(
            lambda m: m.move_to(bars.place_marker(cur()).get_center())
        )
        self.play(FadeIn(readout), *cap("β = 0 is the raw policy"), run_time=0.6)
        cap.hold()
        self.play(
            *cap(
                "A little β moves weight to good moves; the human's move gets likelier"
            ),
            run_time=0.6,
        )
        self.play(beta.animate.set_value(S["beta"]), run_time=2.5)
        cap.hold()
        self.play(*cap("Too much β just plays the best-valued move"), run_time=0.6)
        self.play(beta.animate.set_value(S["beta_engine"]), run_time=2.0)
        cap.hold()
        self.play(
            *cap(
                "So each cell gets its own β, fit so Allie matches the humans' strength"
            ),
            run_time=0.6,
        )
        self.play(beta.animate.set_value(S["beta"]), run_time=1.5)
        cap.hold(0.4)
        bars.bars.clear_updaters()
        bars.marker.clear_updaters()
        self.play(*wipe(self, cap), run_time=0.8)
        self.wait(0.2)


class Trees(Scene):
    """Coverage and lookahead growing side by side from the same root moves (schematic)."""

    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        k = kicker("Two searchers")
        p, q = np.array(S["tree_p"]), np.array(S["tree_q"])
        n = len(p)
        ys = [1.55, 0.55, -0.45, -1.4, -2.3]
        rng = np.random.default_rng(7)

        def panel(cx, color, name, rule):
            xs = cx + np.linspace(-2.75, 2.75, n)
            root = Circle(
                radius=0.17, fill_color=INK, fill_opacity=1, stroke_width=0
            ).move_to([cx, ys[0], 0])
            head = text(name, 30, color, weight="SEMIBOLD").move_to([cx, 2.95, 0])
            sub = text(rule, 18, MUTED).next_to(head, DOWN, buff=0.1)
            return xs, root, VGroup(head, sub)

        cxs, croot, chead = panel(
            -3.6,
            BLUE,
            "Coverage",
            "Allie's original: spread visits over plausible moves",
        )
        lxs, lroot, lhead = panel(
            3.6, ORANGE, "Lookahead", "score every move, then extend the best lines"
        )
        sep = Line([0, 2.6, 0], [0, -2.5, 0], stroke_color=PANEL, stroke_width=3)
        chol = [hollow(x, ys[1], 0.12) for x in cxs]
        lhol = [hollow(x, ys[1], 0.12) for x in lxs]
        ce1 = [edge(croot, h, dashed=True) for h in chol]
        le1 = [edge(lroot, h, dashed=True) for h in lhol]
        ccount = text("0 simulations", 20, BLUE, font=MONO).move_to([-3.6, 2.12, 0])
        lcount = text("0 calls", 20, ORANGE, font=MONO).move_to([3.6, 2.12, 0])
        self.play(
            FadeIn(k),
            FadeIn(schematic()),
            FadeIn(chead),
            FadeIn(lhead),
            Create(sep),
            GrowFromCenter(croot),
            GrowFromCenter(lroot),
            run_time=0.8,
        )
        self.play(
            *[Create(e) for e in (*ce1, *le1)],
            *[FadeIn(h) for h in (*chol, *lhol)],
            FadeIn(ccount),
            FadeIn(lcount),
            *cap("Two searchers grow trees from the same prediction"),
            run_time=0.9,
        )
        cap.hold()

        def grow(nodes, parents, paths, xs0, color, r=(0.12, 0.06, 0.06, 0.05)):
            """Adds a node per path (root move, reply indices...) under its parent; returns new mobjects."""
            new = []
            off = {1: (-0.21, 0.21, -0.07, 0.07), 2: (-0.08, 0.08), 3: (-0.05, 0.05)}
            for path in paths:
                par = nodes[path[:-1]]
                d = len(path) - 1
                x = par.get_x() + off[d][path[-1]]
                c = node(x, ys[d + 1], q[path[0]] + rng.normal(0, 0.22), r[d], color)
                nodes[path] = c
                new += [
                    Line(
                        par.get_center(),
                        c.get_center(),
                        buff=par.radius,
                        stroke_color=RULE,
                        stroke_width=2,
                    ),
                    c,
                ]
            return new

        cn = {(i,): None for i in range(n)}
        ln = {(i,): None for i in range(n)}
        cfill = [node(x, ys[1], v, 0.12, BLUE) for x, v in zip(cxs, q)]
        lfill = [node(x, ys[1], v, 0.12, ORANGE) for x, v in zip(lxs, q)]
        plaus = range(5)
        for i in range(n):
            ln[(i,)] = lfill[i]
            cn[(i,)] = cfill[i] if i in plaus else chol[i]
        ce1s = [
            Line(
                croot.get_center(),
                cfill[i].get_center(),
                buff=0.17,
                stroke_color=RULE,
                stroke_width=2,
            )
            for i in plaus
        ]
        le1s = [
            Line(
                lroot.get_center(),
                f.get_center(),
                buff=0.17,
                stroke_color=RULE,
                stroke_width=2,
            )
            for f in lfill
        ]

        counters = {BLUE: ccount, ORANGE: lcount}

        def count(color, s):
            old = counters[color]
            counters[color] = text(s, 20, color, font=MONO).move_to(old)
            return swap(old, counters[color])

        self.play(
            *[ReplacementTransform(chol[i], cfill[i]) for i in plaus],
            *[ReplacementTransform(ce1[i], ce1s[j]) for j, i in enumerate(plaus)],
            *[ReplacementTransform(lhol[i], lfill[i]) for i in range(n)],
            *[ReplacementTransform(le1[i], le1s[i]) for i in range(n)],
            count(BLUE, "8 simulations"),
            count(ORANGE, "call 1"),
            *cap(
                "Coverage visits the plausible moves; lookahead scores every move in one call"
            ),
            run_time=1.2,
        )
        cap.hold()
        top = sorted(np.argsort(-(np.log(p) + 4 * q), kind="stable")[:8])
        rest = [i for i in range(n) if i not in top]
        rings = [
            Circle(radius=0.19, stroke_color=INK, stroke_width=2.5).move_to(lfill[i])
            for i in top
        ]
        self.play(
            *[GrowFromCenter(r) for r in rings],
            *[m.animate.set_opacity(0.3) for i in rest for m in (lfill[i], le1s[i])],
            *cap("Lookahead keeps the 8 most promising lines, ranked by log p + 4Q"),
            run_time=0.9,
        )
        cap.hold()
        c2 = grow(cn, None, [(i, j) for i in plaus for j in (0, 1)], cxs, BLUE)
        l2 = grow(ln, None, [(i, j) for i in top for j in (0, 1)], lxs, ORANGE)
        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in c2], lag_ratio=0.02),
            LaggedStart(*[GrowFromCenter(m) for m in l2], lag_ratio=0.02),
            count(BLUE, "32 simulations"),
            count(ORANGE, "call 2"),
            *cap(
                "Coverage digs under each plausible move; lookahead adds 2 replies per line"
            ),
            run_time=1.3,
        )
        cap.hold()
        deep = [i for i in top if q[i] > -0.1]
        wide = [i for i in top if i not in deep]
        c3 = grow(
            cn,
            None,
            [(i, 0, j) for i in plaus for j in (0, 1)] + [(i, 1, 0) for i in plaus],
            cxs,
            BLUE,
        )
        l3 = grow(
            ln,
            None,
            [(i, 0, j) for i in deep for j in (0, 1)]
            + [(i, j) for i in wide for j in (2, 3)],
            lxs,
            ORANGE,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in c3], lag_ratio=0.015),
            LaggedStart(*[GrowFromCenter(m) for m in l3], lag_ratio=0.015),
            count(BLUE, "128 simulations"),
            count(ORANGE, "call 3"),
            *cap(
                "Lookahead goes deeper where one reply stands out, and wider where it doesn't"
            ),
            run_time=1.3,
        )
        c4 = grow(
            cn,
            None,
            [(i, 0, 0, j) for i in (0, 2) for j in (0, 1)]
            + [(i, 0, 1, 0) for i in plaus],
            cxs,
            BLUE,
        )
        l4 = grow(
            ln, None, [(i, 0, 0, j) for i in deep[:4] for j in (0, 1)], lxs, ORANGE
        )
        self.play(
            LaggedStart(*[GrowFromCenter(m) for m in c4], lag_ratio=0.02),
            LaggedStart(*[GrowFromCenter(m) for m in l4], lag_ratio=0.02),
            count(BLUE, "256 simulations"),
            count(ORANGE, "call 4"),
            run_time=1.0,
        )
        cap.hold()

        def flash(mobs, color):
            lines = [m for m in mobs if isinstance(m, Line)]
            ups = [
                Line(e.get_end(), e.get_start(), stroke_color=color, stroke_width=8)
                for e in lines
            ]
            return [ShowPassingFlash(u, time_width=0.6) for u in ups]

        shift = rng.normal(0, 0.2, n)
        self.play(
            *flash([*c2, *c3, *c4, *ce1s], BLUE),
            *flash([*l2, *l3, *l4, *le1s], ORANGE),
            *[
                cfill[i].animate.set_fill(
                    node(0, 0, q[i] + shift[i], 0.1, BLUE).get_fill_color()
                )
                for i in plaus
            ],
            *[
                lfill[i].animate.set_fill(
                    node(0, 0, q[i] + shift[i], 0.1, ORANGE).get_fill_color()
                )
                for i in top
            ],
            *cap("Both back the values up to the root moves, then apply the same tilt"),
            run_time=1.4,
        )
        cap.hold()
        self.play(*wipe(self, keep=[k, cap.cur]), run_time=0.8)

        v = D["versus"]
        cards = VGroup()
        for color, big, line1, line2 in (
            (
                BLUE,
                f"{v['coverage_as_good']} of {v['cells']}",
                "cells where coverage gets as close",
                "or closer to human strength",
            ),
            (
                ORANGE,
                f"{v['lookahead_cheaper']} of {v['cells']}",
                "cells where lookahead is within",
                "20 Elo and cheaper",
            ),
        ):
            body = VGroup(
                text(big, 64, color, font=MONO, weight="MEDIUM"),
                text(line1, 24, INK2),
                text(line2, 24, INK2),
            ).arrange(DOWN, buff=0.16)
            box = RoundedRectangle(
                corner_radius=0.16,
                width=5.6,
                height=3.0,
                fill_color=OFF,
                fill_opacity=1,
                stroke_color=color,
                stroke_width=3,
            )
            cards.add(VGroup(box, body.move_to(box)))
        cards.arrange(RIGHT, buff=0.8).move_to([0, 0.2, 0])
        self.play(
            LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in cards], lag_ratio=0.3),
            *cap(
                f"On the {v['cells']} searching cells they tie, so coverage, already in the bot, ships"
            ),
            run_time=1.2,
        )
        cap.hold(1.0)
        self.play(*wipe(self, cap), run_time=0.8)
        self.wait(0.2)


class Results(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        k = kicker("Results per cell")
        labels = grid_labels()
        idx = [(r, c) for r in range(len(FMTS)) for c in range(len(BINS))]
        key = lambda r, c: f"{FMTS[r]}/{BINS[c]}"  # noqa: E731

        def raw_tile(r, c):
            e = CELLS[key(r, c)]["raw_elo"]
            fill, ink = heat(e)
            return tile(r, c, num(e), fill, ink)

        def ship_tile(r, c):
            v = CELLS[key(r, c)]
            if v["choice"] == "raw":
                fill, ink = heat(v["elo"])
                return tile(r, c, num(v["elo"]), fill, ink).set_opacity(0.35)
            fill, ink = heat(v["elo"])
            return tile(
                r, c, num(v["elo"]), fill, ink, sub=f"cov {v['sims']}", stroke=BLUE
            )

        raws = VGroup(*[raw_tile(r, c) for r, c in idx])
        unit = text(
            "Elo vs the humans in each cell", 22, INK2, weight="SEMIBOLD"
        ).move_to([X0 - CW / 2, -1.85, 0], aligned_edge=LEFT)
        self.play(
            FadeIn(k),
            FadeIn(labels),
            FadeIn(unit),
            *cap("Before: the raw policy in every cell, in Elo against the humans"),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(t, scale=0.9) for t in raws], lag_ratio=0.015),
            run_time=1.6,
        )
        cap.hold()
        search = [
            i for i, (r, c) in enumerate(idx) if CELLS[key(r, c)]["choice"] != "raw"
        ]
        ships = VGroup(*[ship_tile(r, c) for r, c in idx])
        self.play(
            LaggedStart(*[swap(raws[i], ships[i]) for i in search], lag_ratio=0.06),
            *cap(
                f"After: {D['n_search']} cells search, all with coverage, at 32 to 256 simulations"
            ),
            run_time=2.4,
        )
        cap.hold()
        lo, hi = D["ce_range"]
        self.play(
            *cap(
                f"Each predicts human moves better than raw: CE {num(lo, '{:+.3f}')} to {num(hi, '{:+.3f}')} nats"
            ),
            run_time=0.6,
        )
        cap.hold()
        rest = [i for i in range(len(idx)) if i not in search]
        self.play(
            *[swap(raws[i], ships[i]) for i in rest],
            *cap(
                "No search in bullet; at lower ratings none clears the bar, or raw is close"
            ),
            run_time=1.0,
        )
        cap.hold()

        rows = VGroup()
        kx = 0.0105
        anims = []
        for j, f in enumerate(("blitz", "rapid", "classical")):
            y = -1.95 - 0.32 * j
            name = text(f, 20, INK2).move_to([-4.0, y, 0], aligned_edge=RIGHT)
            a, b = D["raw"][f]["rms"], D["shipped"][f]["rms"]
            bar = Rectangle(
                width=kx * a,
                height=0.22,
                fill_color=RULE,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([-3.8, y, 0], aligned_edge=LEFT)
            val = text(f"{a}", 20, INK2, font=MONO).next_to(bar, RIGHT, buff=0.12)
            rows.add(VGroup(name, bar, val))
            nb = Rectangle(
                width=kx * b,
                height=0.22,
                fill_color=BLUE,
                fill_opacity=1,
                stroke_width=0,
            ).move_to([-3.8, y, 0], aligned_edge=LEFT)
            anims += [
                Transform(bar, nb),
                swap(
                    val,
                    text(f"{a} → {b}", 20, BLUE, font=MONO, weight="MEDIUM").next_to(
                        nb, RIGHT, buff=0.12
                    ),
                ),
            ]
        rt = text(
            "typical miss, Elo (RMS over cells)", 20, INK2, weight="SEMIBOLD"
        ).move_to([X0 - CW / 2, -1.62, 0], aligned_edge=LEFT)
        self.play(FadeOut(unit), FadeIn(rows), FadeIn(rt), run_time=0.6)
        rr = [D["raw"][f]["rms"] for f in ("blitz", "rapid", "classical")]
        ss = [D["shipped"][f]["rms"] for f in ("blitz", "rapid", "classical")]
        self.play(
            *anims,
            *cap(
                f"The typical miss falls from {min(rr)}–{max(rr)} Elo to {min(ss)}–{max(ss)}"
            ),
            run_time=1.5,
        )
        cap.hold()
        weak = [
            FMTS.index("classical") * len(BINS) + BINS.index(b) for b in (2400, 2600)
        ]
        ring = RoundedRectangle(
            corner_radius=0.1,
            width=2 * CW + GAP + 0.2,
            height=CH + 0.2,
            stroke_color=ORANGE,
            stroke_width=5,
        ).move_to(VGroup(*[ships[i] for i in weak]))
        e1, e2 = (CELLS[f"classical/{b}"]["elo"] for b in (2400, 2600))
        self.play(
            Create(ring),
            *cap(
                f"Still short: classical 2400 and 2600, at {num(e1)} and {num(e2)} Elo"
            ),
            run_time=0.8,
        )
        cap.hold(0.6)
        self.play(*wipe(self, cap), run_time=0.8)
        self.wait(0.2)


class Ceiling(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        k = kicker("The ceiling")
        C = D["ceiling"]
        e1, e2 = (CELLS[f"classical/{b}"]["elo"] for b in (2400, 2600))
        cards = VGroup()
        for b, e in ((2400, e1), (2600, e2)):
            fill, ink = heat(e)
            box = RoundedRectangle(
                corner_radius=0.12,
                width=3.0,
                height=1.7,
                fill_color=fill,
                fill_opacity=1,
                stroke_width=0,
            )
            body = VGroup(
                text(f"classical {b}", 24, ink),
                text(num(e), 48, ink, font=MONO, weight="MEDIUM"),
            ).arrange(DOWN, buff=0.12)
            cards.add(VGroup(box, body.move_to(box)))
        cards.arrange(RIGHT, buff=0.6).move_to([0, 0.4, 0])
        self.play(
            FadeIn(k),
            FadeIn(cards, shift=0.2 * UP),
            *cap(
                "Why do classical 2400 and 2600 stay short: too little search, or bad values?"
            ),
            run_time=0.9,
        )
        cap.hold()
        self.play(FadeOut(cards), run_time=0.5)

        yard = VGroup()
        for color, title, big, l1, l2 in (
            (
                INK2,
                "Stockfish's values",
                f"{C['stockfish_cells']} of {C['stockfish_cells']}",
                "strong cells reach human strength",
                "with CE below the raw policy",
            ),
            (
                BLUE,
                "Allie's own values",
                f"{num(C['plateau_elo'][0])} to {num(C['plateau_elo'][1])}",
                "Elo short in strong classical,",
                "at the best-predicting tilt",
            ),
        ):
            body = VGroup(
                text(title, 26, color, weight="SEMIBOLD"),
                text(big, 54, color, font=MONO, weight="MEDIUM"),
                text(l1, 22, INK2),
                text(l2, 22, INK2),
            ).arrange(DOWN, buff=0.14)
            box = RoundedRectangle(
                corner_radius=0.16,
                width=5.8,
                height=3.2,
                fill_color=OFF,
                fill_opacity=1,
                stroke_color=color,
                stroke_width=3,
            )
            yard.add(VGroup(box, body.move_to(box)))
        yard.arrange(RIGHT, buff=0.7).move_to([0, 0.3, 0])
        self.play(
            FadeIn(yard[0], shift=0.2 * UP),
            *cap(
                "Yardstick: with Stockfish's values as Q, the same tilt reaches human strength"
            ),
            run_time=0.9,
        )
        cap.hold()
        self.play(
            FadeIn(yard[1], shift=0.2 * UP),
            *cap("With Allie's own values, strong classical stalls"),
            run_time=0.9,
        )
        cap.hold()
        self.play(
            *cap("The search and the tilt work. Allie's value estimate is the limit")
        )
        cap.hold()
        self.play(FadeOut(yard), run_time=0.6)

        # value noise, schematic: noisy Q under a big beta makes the distribution jump
        np_, nq, sd, nb = (
            np.array(S["noise_p"]),
            np.array(S["noise_q"]),
            S["noise_sd"],
            S["noise_beta"],
        )
        m = len(np_)
        xs = np.linspace(-4.2, 2.2, m)
        Qp = Plot((0, 1), (-0.6, 0.6), (-5.0, 3.0, 0.15, 2.25))
        base, scale = -2.55, 2.6
        qaxis = VGroup(
            Line(Qp.p(0, 0), Qp.p(1, 0), stroke_color=PANEL, stroke_width=2),
            text("Q", 22, BLUE, font=MONO, weight="MEDIUM").move_to([-5.3, Qp.y(0), 0]),
            text("π", 22, ORANGE, font=MONO, weight="MEDIUM").move_to(
                [-5.3, base + 0.6, 0]
            ),
            Line([-4.75, base, 0], [2.65, base, 0], stroke_color=RULE, stroke_width=2),
        )

        def band(s):
            return VGroup(
                *[
                    RoundedRectangle(
                        corner_radius=0.06,
                        width=0.34,
                        height=4 * s * (Qp.box[3] - Qp.box[2]) / 1.2,
                        fill_color=PALE_BLUE,
                        fill_opacity=0.35,
                        stroke_width=0,
                    ).move_to([x, Qp.y(v), 0])
                    for x, v in zip(xs, nq)
                ]
            )

        def draw(s, seed):
            z = nq + np.random.default_rng(seed).normal(0, s, m)
            dots = VGroup(
                *[Dot([x, Qp.y(v), 0], radius=0.09, color=BLUE) for x, v in zip(xs, z)]
            )
            pi = tilt(np_, z, nb)
            bars = VGroup(
                *[
                    Rectangle(
                        width=0.5,
                        height=max(v * scale, 0.01),
                        fill_color=ORANGE,
                        fill_opacity=1,
                        stroke_width=0,
                    ).move_to([x, base, 0], aligned_edge=DOWN)
                    for x, v in zip(xs, pi)
                ]
            )
            return VGroup(dots, bars)

        tag = text(f"β = {nb}", 28, ORANGE, font=MONO, weight="MEDIUM").move_to(
            [4.4, 1.2, 0]
        )
        views = text("1 view", 26, BLUE, weight="SEMIBOLD").move_to([4.4, 0.4, 0])
        stag = VGroup(
            schematic(),
            VGroup(
                RoundedRectangle(
                    corner_radius=0.04,
                    width=0.3,
                    height=0.3,
                    fill_color=PALE_BLUE,
                    fill_opacity=0.35,
                    stroke_width=0,
                ),
                text("spread of Q", 20, MUTED),
            )
            .arrange(RIGHT, buff=0.12)
            .next_to(views, DOWN, buff=0.35, aligned_edge=LEFT),
        )
        bd = band(sd)
        cur = draw(sd, 0)
        self.play(
            FadeIn(qaxis),
            FadeIn(bd),
            FadeIn(cur),
            FadeIn(tag),
            FadeIn(views),
            FadeIn(stag),
            *cap(
                "Allie's Q is a noisy estimate, and strong classical needs a big β (8 to 12)"
            ),
            run_time=0.9,
        )
        cap.hold(0.2)
        self.play(
            *cap(
                "A big β turns small errors in Q into big swings in which move gets played"
            ),
            run_time=0.5,
        )
        for seed in (1, 2, 3, 4):
            self.play(Transform(cur, draw(sd, seed)), run_time=0.55)
            self.wait(0.15)
        cap.hold()
        s2 = sd / np.sqrt(2)
        self.play(
            Transform(bd, band(s2)),
            Transform(cur, draw(s2, 5)),
            swap(
                views,
                views2 := text("2 views", 26, BLUE, weight="SEMIBOLD").move_to(views),
            ),
            *cap(
                "Average two views (true ratings, 2800 header): part of the noise cancels"
            ),
            run_time=0.9,
        )
        for seed in (6, 7, 8, 9):
            self.play(Transform(cur, draw(s2, seed)), run_time=0.55)
            self.wait(0.15)
        cap.hold()
        self.play(FadeOut(VGroup(qaxis, bd, cur, tag, views2, stag)), run_time=0.6)

        P = Plot((32, 4096), (-0.1, 0.45), (-4.7, 4.3, -2.05, 2.1), xlog=True)
        fr = P.frame(
            [32, 64, 128, 256, 512, 1024, 2048, 4096],
            [-0.1, 0, 0.1, 0.2, 0.3, 0.4],
            xfmt=lambda v: f"{v:,}",
            yfmt=lambda v: num(v, "{:+.1f}") if v else "0",
            xlabel="network evaluations per move",
            ylabel="classical 2400: CE above raw at human strength, nats",
        )
        bar = Line(P.p(32, 0), P.p(4096, 0), stroke_color=INK, stroke_width=4)
        barl = text("the bar", 20, INK, weight="SEMIBOLD").next_to(
            bar, RIGHT, buff=0.15
        )
        sf = DashedLine(
            P.p(32, C["stockfish_ce"]),
            P.p(4096, C["stockfish_ce"]),
            stroke_color=MUTED,
            stroke_width=3,
            dash_length=0.12,
        )
        sfl = text("Stockfish's values (yardstick)", 18, MUTED).next_to(
            P.p(32, C["stockfish_ce"]), UP + RIGHT, buff=0.08
        )

        def series(rows, color, width=4):
            rows = [r for r in rows if r["ce"] is not None and r["ce"] <= 0.45]
            return P.line(
                [r["leaves"] for r in rows], [r["ce"] for r in rows], color, width
            ), rows

        la, lar = series(C["lookahead"], ORANGE)
        cv, cvr = series(C["coverage"], PALE_BLUE)
        tv, tvr = series(C["two_view"], BLUE)
        lal = (
            text(
                f"lookahead, {lar[0]['budget']} to {lar[-1]['budget']} calls",
                20,
                ORANGE,
                weight="SEMIBOLD",
            )
            .next_to(la[1][-1], UP, buff=0.15)
            .shift(0.4 * LEFT)
        )
        cvl = text("coverage, 1 view", 20, BLUE_TXT, weight="SEMIBOLD").next_to(
            cv[1][1], RIGHT, buff=0.15
        )
        tvl = text("coverage, 2 views", 20, BLUE, weight="SEMIBOLD").next_to(
            tv[1][1], DOWN + LEFT, buff=0.1
        )
        self.play(
            FadeIn(fr),
            Create(bar),
            FadeIn(barl),
            *cap("Classical 2400: CE above raw once search is tuned to human strength"),
            run_time=1.0,
        )
        cap.hold()
        self.play(
            Create(sf),
            FadeIn(sfl),
            *cap("Stockfish's values get there below the bar"),
            run_time=0.9,
        )
        cap.hold()
        self.play(
            Create(la[0]),
            FadeIn(la[1]),
            FadeIn(lal),
            *cap(
                f"Lookahead barely moves from {lar[0]['budget']} to {lar[-1]['budget']} calls: depth doesn't fix noise"
            ),
            run_time=1.4,
        )
        cap.hold()
        self.play(
            Create(cv[0]),
            FadeIn(cv[1]),
            FadeIn(cvl),
            *cap(
                "Coverage averages more evaluations per move as it grows; CE creeps down"
            ),
            run_time=1.6,
        )
        cap.hold()
        one = min(cvr, key=lambda r: abs(r["leaves"] - 1000))
        two = min(tvr, key=lambda r: abs(r["leaves"] - 1000))
        last, secs = tvr[-1], C["seconds"]
        assert one["budget"] == 1024 and two["budget"] == 512 and last["budget"] == 1024

        def ring(r, color):
            return Circle(radius=0.11, stroke_color=color, stroke_width=3).move_to(
                P.p(r["leaves"], r["ce"])
            )

        def note(s, color):
            return text(s, 19, color, font=MONO, weight="MEDIUM")

        notes = VGroup(
            note(
                f"1 view,  1,024 sims  {one['ce']:+.3f} · {secs['coverage 1024']:g} s",
                BLUE_TXT,
            ),
            note(f"2 views,   512 sims  {two['ce']:+.3f}", BLUE),
            note(
                f"2 views, 1,024 sims  {last['ce']:+.3f} · {secs['two-view 1024']:g} s",
                BLUE,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        notes.move_to(P.p(4096, 0.44), aligned_edge=UP + RIGHT).shift(0.15 * RIGHT)
        same = VGroup(
            DashedLine(
                P.p(1000, -0.1),
                P.p(1000, 0.31),
                stroke_color=RULE,
                stroke_width=2,
                dash_length=0.08,
            ),
            ring(one, BLUE_TXT),
            ring(two, BLUE),
        )
        self.play(
            Create(tv[0]),
            FadeIn(tv[1]),
            FadeIn(tvl),
            *cap("Two views beat twice the simulations at the same cost"),
            run_time=1.6,
        )
        self.play(FadeIn(same), FadeIn(notes[0]), FadeIn(notes[1]), run_time=0.8)
        cap.hold()
        self.play(
            Create(ring(last, BLUE)),
            FadeIn(notes[2]),
            *cap(
                f"Two views at {last['budget']:,} simulations reach {last['ce']:+.3f}, at about {secs['two-view 1024']:g} s a move"
            ),
            run_time=0.8,
        )
        cap.hold(0.6)
        self.play(*wipe(self, cap, keep=[k]), run_time=0.8)

        motto = text(
            "Stockfish measures. It never teaches.", 48, ORANGE, weight="SEMIBOLD"
        ).move_to([0, 0.5, 0])
        sub = text(
            "Allie learns only from human games; a better value has to come from them too",
            24,
            INK2,
        ).next_to(motto, DOWN, buff=0.4)
        self.play(FadeIn(motto, shift=0.15 * UP), run_time=0.9)
        self.play(FadeIn(sub), run_time=0.7)
        self.wait(3.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
        self.wait(0.2)


class Live(Scene):
    def construct(self):
        paper(self)
        cap = Captions(self, wps=2.6)
        L = D["live"]
        k = kicker("On Lichess now")
        labels = grid_labels()
        shades = {0: PANEL, 32: "#C6D6E3", 128: PALE_BLUE, 256: BLUE}
        idx = [(r, c) for r in range(len(FMTS)) for c in range(len(BINS))]

        def t(r, c):
            v = CELLS[f"{FMTS[r]}/{BINS[c]}"]
            s = v["sims"]
            return tile(
                r,
                c,
                str(s) if s else "raw",
                shades[s],
                OFF if s == 256 else (MUTED if not s else INK),
            )

        tiles = VGroup(*[t(r, c) for r, c in idx])
        legend = VGroup()
        for s, label in (
            (0, "raw policy, T = 1"),
            (32, "coverage 32"),
            (128, "128"),
            (256, "256 simulations"),
        ):
            sw = RoundedRectangle(
                corner_radius=0.04,
                width=0.34,
                height=0.26,
                fill_color=shades[s],
                fill_opacity=1,
                stroke_width=0,
            )
            legend.add(VGroup(sw, text(label, 20, INK2).next_to(sw, RIGHT, buff=0.12)))
        legend.arrange(RIGHT, buff=0.45).move_to(
            [X0 - CW / 2, -1.9, 0], aligned_edge=LEFT
        )
        self.play(
            FadeIn(k),
            FadeIn(labels),
            *cap(
                "Each game plays in the cell of its time control and the opponent's rating"
            ),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(x, scale=0.9) for x in tiles], lag_ratio=0.015),
            FadeIn(legend),
            run_time=1.6,
        )
        cap.hold()
        self.play(
            *cap(
                f"{D['n_search']} cells search with coverage; bullet and lower ratings play the raw policy"
            ),
            run_time=0.6,
        )
        cap.hold()
        self.play(
            *cap(
                "A search gets a tenth of the clock; past a fifth, Allie plays the policy"
            ),
            run_time=0.6,
        )
        cap.hold()
        self.play(
            *cap(f"{D['model']} on {L['cpus']} CPUs, up to {L['games']} games at once"),
            run_time=0.6,
        )
        cap.hold()
        self.play(FadeOut(VGroup(tiles, labels, legend)), *cap.clear(), run_time=0.8)
        url = text(L["url"], 44, INK, font=MONO, weight="MEDIUM").move_to([0, 0.5, 0])
        pre = text("Play Allie", 28, ORANGE, weight="SEMIBOLD").next_to(
            url, UP, buff=0.35
        )
        repo = text("github.com/y0mingzhang/allie", 24, MUTED, font=MONO).next_to(
            url, DOWN, buff=0.45
        )
        self.play(FadeIn(pre), FadeIn(url, shift=0.15 * UP), run_time=0.9)
        self.play(FadeIn(repo), run_time=0.6)
        self.wait(3.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.9)
        self.wait(0.3)
