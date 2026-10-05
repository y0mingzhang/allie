"""Fitting Allie's scaling law: what an isoFLOP slice is, why its trough moves with compute, Chinchilla's three
approaches on the MoE sweep, the Allie 2.0 miss, the data term, why the law missed and the Allie 2.1 forecast.
Every number drawn is in data.json["scaling"]."""

import numpy as np
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Axes,
    Circle,
    Create,
    DashedLine,
    DashedVMobject,
    Dot,
    DoubleArrow,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    ManimColor,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Star,
    Transform,
    TransformFromCopy,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    linear,
    smooth,
)

from style import (
    BG,
    BLUE,
    DATA,
    INK,
    INK2,
    MUTED,
    ORANGE,
    PANEL,
    RULE,
    Captions,
    kicker,
    paper,
    text,
)

D = DATA["scaling"]
M, SW, LAW, A1, A2 = D["meta"], D["sweep"], D["law"], D["envelope"], D["troughs"]
BUD = ("#D98A55", ORANGE, "#7A2E0C")
GRIDC, PALE, GREY = "#E2DCCF", "#F2D3BF", "#9C968C"

VIDEO = dict(
    name="scaling-law",
    scenes=[*[f"S{i}" for i in range(9)], "M1", "M2", "M3", "S9", "S10"],
    gif=(["S7", 8.6], 13.5),
    poster=["S10", 27.0],
)


class Cap(Captions):
    def __init__(self, scene):
        super().__init__(scene, wps=2.6)

    def make(self, s, markup):
        t = super().make(s, markup)
        if t.height > 1.5 * text("Hx", self.px).height:
            raise ValueError(f"caption wraps: {s}")
        return t


def mix(a, b, t):
    return ManimColor(a).interpolate(ManimColor(b), t)


def kap(n):
    return np.interp(np.log(n), M["kappa_lnn"], M["kappa"])


def law(n, d, p=LAW):
    return p["E"] + p["A"] * (n / 1e7) ** -p["alpha"] + p["B"] * (d / 1e8) ** -p["beta"]


def iso(n, c, p=LAW):
    return law(n, c / (kap(n) * n), p)


def sci(v, d=1):
    e = int(np.floor(np.log10(v) + 1e-9))
    return f"{v / 10**e:.{d}f}×10<sup>{e}</sup>"


def size(n):
    return f"{n / 1e6:.0f}M" if n < 1e9 else f"{n / 1e9:.1f}B"


def lab(s, px=24, color=INK2, weight="MEDIUM"):
    return text(s, px, color, weight=weight, markup=True)


def outro(scene, t=0.9):
    for m in scene.mobjects:
        m.clear_updaters()
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=t)
    scene.wait(0.2)


class Plot(VGroup):
    """Axes over data ranges xr, yr (log10 when logx / logy) with grid, tick labels and titles."""

    def __init__(
        self,
        xr,
        yr,
        w,
        h,
        center,
        xt=(),
        yt=(),
        xl="",
        yl="",
        logx=True,
        logy=False,
        px=22,
    ):
        super().__init__()
        self.logx, self.logy = logx, logy
        f = lambda v, lg: np.log10(v) if lg else v  # noqa: E731
        self.x0, self.x1 = f(xr[0], logx), f(xr[1], logx)
        self.y0, self.y1 = f(yr[0], logy), f(yr[1], logy)
        self.ax = Axes(
            x_range=[self.x0, self.x1, self.x1 - self.x0],
            y_range=[self.y0, self.y1, self.y1 - self.y0],
            x_length=w,
            y_length=h,
            tips=False,
            axis_config=dict(stroke_color=RULE, stroke_width=2, include_ticks=False),
        ).move_to(np.array([*center, 0][:3], float))
        self.ax.y_axis.set_opacity(0)
        self.ax.x_axis.set_opacity(0)
        self.base = Line(
            self.ax.c2p(self.x0, self.y0),
            self.ax.c2p(self.x1, self.y0),
            stroke_color=RULE,
            stroke_width=2,
        )
        self.ax.add(self.base)
        self.grid = VGroup(
            *[
                Line(
                    self.p(xr[0], y),
                    self.p(xr[1], y),
                    stroke_color=GRIDC,
                    stroke_width=1.6,
                )
                for y, _ in yt
            ]
        )
        self.yt = VGroup(
            *[
                lab(s, px, MUTED, "NORMAL").next_to(self.p(xr[0], y), LEFT, buff=0.15)
                for y, s in yt
            ]
        )
        self.xt = VGroup(
            *[
                lab(s, px, MUTED, "NORMAL").next_to(self.p(x, yr[0]), DOWN, buff=0.15)
                for x, s in xt
            ]
        )
        self.add(self.grid, self.ax, self.yt, self.xt)
        if xl:
            self.xl = lab(xl, px + 2, INK2).next_to(
                self.xt if xt else self.base, DOWN, buff=0.12 if xt else 0.15
            )
            self.add(self.xl)
        if yl:
            self.yl = lab(yl, px + 2, INK2).next_to(
                self.ax.c2p(self.x0, self.y1), UP, buff=0.3
            )
            self.yl.align_to(self.yt if yt else self.ax, LEFT)
            self.add(self.yl)

    def t(self, x, y):
        return (np.log10(x) if self.logx else x), (np.log10(y) if self.logy else y)

    def p(self, x, y):
        return self.ax.c2p(*self.t(x, y))

    def inside(self, y):
        return self.y0 <= self.t(1, y)[1] <= self.y1

    def line(self, xs, ys, color=INK, width=4, dash=0, **kw):
        """Polyline through data points, clipped to the y range."""
        tx, ty = self.t(np.asarray(xs, float), np.asarray(ys, float))
        tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
        ok = (ty >= self.y0) & (ty <= self.y1)
        vm = VMobject(stroke_color=color, stroke_width=width, **kw)
        if not ok.any():
            return vm.set_points_as_corners([self.ax.c2p(tx[0], self.y1)] * 2)
        i, j = np.flatnonzero(ok)[[0, -1]]
        px, py = list(tx[i : j + 1]), list(ty[i : j + 1])
        for k, at in ((i - 1, 0), (j + 1, len(px) + 1)):
            if 0 <= k < len(tx):
                b = self.y1 if ty[k] > self.y1 else self.y0
                a, c = (k, k + 1) if at == 0 else (k - 1, k)
                u = (b - ty[a]) / (ty[c] - ty[a])
                px.insert(at, tx[a] + u * (tx[c] - tx[a]))
                py.insert(at, b)
        vm.set_points_as_corners([self.ax.c2p(x, y) for x, y in zip(px, py)])
        return DashedVMobject(vm, num_dashes=dash, dashed_ratio=0.55) if dash else vm

    def hline(self, y, color=RULE, width=2.5):
        x0, x1 = (10**self.x0, 10**self.x1) if self.logx else (self.x0, self.x1)
        return DashedLine(
            self.p(x0, y),
            self.p(x1, y),
            color=color,
            stroke_width=width,
            dash_length=0.12,
        )

    def vline(self, x, color=RULE, width=2.5, y0=None, y1=None):
        lo = y0 if y0 is not None else (10**self.y0 if self.logy else self.y0)
        hi = y1 if y1 is not None else (10**self.y1 if self.logy else self.y1)
        return DashedLine(
            self.p(x, lo),
            self.p(x, hi),
            color=color,
            stroke_width=width,
            dash_length=0.1,
        )


def dot(pos, color, r=0.085, hollow=False):
    d = Dot(
        pos,
        radius=r,
        color=color,
        stroke_width=2.5,
        stroke_color=color if hollow else BG,
    )
    return d.set_fill(BG if hollow else color, 1)


def ring(pos, r=0.2):
    return Circle(radius=r, stroke_color=INK, stroke_width=4).move_to(pos)


def diamond(pos, color, r=0.1):
    return (
        Square(r * 1.42, color=INK, stroke_width=2, fill_color=color, fill_opacity=1)
        .rotate(PI / 4)
        .move_to(pos)
    )


def star(pos, r=0.24):
    return Star(
        5,
        outer_radius=r,
        inner_radius=r * 0.45,
        color=INK,
        fill_opacity=1,
        stroke_width=0,
    ).move_to(pos)


def lossplot(center=(0.3, 0.05), w=10.0, h=4.7, yr=(1.30, 1.54), px=22):
    ys = np.arange(np.ceil(yr[0] / 0.05 - 1e-6) * 0.05, yr[1] + 1e-9, 0.05)
    return Plot((1e7, 4.2e8), yr, w, h, center, xt=[(1e7, "10M"), (3e7, "30M"), (1e8, "100M"), (3e8, "300M")],
                yt=[(y, f"{y:.2f}") for y in ys], xl="active parameters N", yl="loss (nats)", px=px)  # fmt: skip


def points(P, b, hollow_out=False, r=0.085):
    return VGroup(*[dot(P.p(q["n"], q["loss"]), BUD[b], r, hollow_out and not q["window"])
                    for q in SW[b]["points"] if P.inside(q["loss"])])  # fmt: skip


def parabola(P, b, pad=0.15, width=5):
    s = SW[b]
    ns = [q["n"] for q in s["points"] if q["window"]]
    x = np.logspace(
        np.log10(min(ns)) - pad, min(np.log10(max(ns)) + pad, np.log10(4.2e8)), 60
    )
    return P.line(x, np.polyval(s["parabola"], np.log(x)), BUD[b], width)


def trough(P, b, side=DOWN):
    s = SW[b]
    r = ring(P.p(s["n_opt"], s["l"]))
    return r, lab(size(s["n_opt"]), 24, INK, "SEMIBOLD").next_to(r, side, buff=0.1)


def isocurve(P, c, p=LAW, color=INK, width=4):
    n = np.logspace(np.log10(1.2e7), np.log10(4.2e8), 120)
    return P.line(n, iso(n, c, p), color, width)


def legend(at=(0.3, 2.95)):
    g = VGroup()
    for b, s in enumerate(SW):
        g.add(
            VGroup(
                Dot(radius=0.09, color=BUD[b]),
                lab(sci(s["c"]) + (" FLOPs" if b == 2 else ""), 22, INK),
            ).arrange(RIGHT, buff=0.12)
        )
    return g.arrange(RIGHT, buff=0.5).move_to([*at, 0])


def tabs(i, ref):
    names = ("1  Envelope", "2  Troughs", "3  Parametric")
    g = VGroup(
        *[
            text(
                s,
                24,
                INK if j == i else MUTED,
                weight="SEMIBOLD" if j == i else "MEDIUM",
            )
            for j, s in enumerate(names)
        ]
    )
    g.arrange(RIGHT, buff=0.8).move_to([2.2, ref.get_center()[1], 0])
    u = Line(
        g[i].get_corner(DOWN + LEFT),
        g[i].get_corner(DOWN + RIGHT),
        color=ORANGE,
        stroke_width=4,
    )
    return VGroup(g, u.shift(0.1 * DOWN))


def pulse(m):
    return Indicate(m, color=ORANGE, scale_factor=1.15)


# ------------------------------------------------------------------ title


class S0(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        P = lossplot(center=(0, 0.2), w=12, h=6)
        ghosts = VGroup(
            *[parabola(P, b, 0.35, 6).set_stroke(opacity=0.22) for b in range(3)]
        )
        title = text("Fitting Allie's Scaling Law", 66, INK, weight="SEMIBOLD")
        sub = text(
            f"{M['runs']} training runs, three ways to fit them, one forecast", 30, INK2
        )
        VGroup(title, sub).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(
            [0, 0.3, 0]
        )
        bar = (
            Line(LEFT * 0.65, RIGHT * 0.65, color=ORANGE, stroke_width=6)
            .next_to(title, UP, buff=0.35)
            .align_to(title, LEFT)
        )
        k = kicker("Allie · scaling law")
        self.play(
            LaggedStart(*[Create(g) for g in ghosts], lag_ratio=0.25),
            FadeIn(k),
            run_time=1.6,
        )
        self.play(GrowFromEdge(bar, LEFT), Write(title), run_time=1.3)
        self.play(
            FadeIn(sub, shift=0.15 * UP),
            *cap("Given a compute budget, how big should the model be?"),
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 1. the isoFLOP slice


class S1(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        s = SW[0]
        c = s["c"]
        k = kicker("The isoFLOP slice")
        f = lab("C ≈ 6 · N · D", 38, INK, "SEMIBOLD").move_to([-4.35, 2.55, 0])
        fsub = lab(f"compute fixed at {sci(c)} FLOPs", 22, INK2).next_to(
            f, DOWN, buff=0.18
        )
        self.play(
            FadeIn(k),
            Write(f),
            *cap("Training compute is about 6 × parameters N × tokens D."),
        )
        cap.hold()

        nmax = max(q["n"] for q in s["points"])
        nmin = min(q["n"] for q in s["points"])
        dn = lambda n: c / (kap(n) * n)  # noqa: E731
        sx, sy = 3.9 / nmax, 3.7 / dn(nmin)
        o = np.array([-6.55, -2.25, 0])
        ln = ValueTracker(np.log10(28e6))

        def box():
            n = 10 ** ln.get_value()
            r = Rectangle(
                width=sx * n,
                height=sy * dn(n),
                stroke_color=ORANGE,
                stroke_width=4,
                fill_color=ORANGE,
                fill_opacity=0.14,
            )
            return r.move_to(o, aligned_edge=DOWN + LEFT)

        rect = always_redraw(box)
        nl = always_redraw(
            lambda: lab(
                f"N = {size(10 ** ln.get_value())} parameters", 24, INK
            ).next_to(o, DOWN, buff=0.18, aligned_edge=LEFT)
        )
        dl = always_redraw(
            lambda: lab(
                f"D = {dn(10 ** ln.get_value()) / 1e9:.1f}B tokens", 24, INK
            ).next_to(rect, UP, buff=0.14, aligned_edge=LEFT)
        )
        self.play(
            FadeIn(fsub),
            GrowFromEdge(rect, DOWN),
            FadeIn(nl),
            FadeIn(dl),
            *cap("Fix the compute, and a bigger model must see fewer tokens."),
        )
        self.play(ln.animate.set_value(np.log10(nmax)), run_time=1.6)
        self.play(ln.animate.set_value(np.log10(nmin)), run_time=1.6)
        cap.hold()

        P = lossplot(center=(2.95, 0.0), w=7.0, h=4.6)
        cur = always_redraw(lambda: P.vline(10 ** ln.get_value(), MUTED, 2))
        self.play(
            FadeIn(P),
            FadeIn(cur),
            *cap(
                "Train one model at each split and measure its loss: an <i>isoFLOP slice</i>."
            ),
        )
        dots = points(P, 0, r=0.1)
        order = sorted(range(len(dots)), key=lambda i: s["points"][i]["n"])
        for i in order:
            self.play(
                ln.animate.set_value(np.log10(s["points"][i]["n"])),
                run_time=0.55,
                rate_func=smooth,
            )
            self.play(GrowFromCenter(dots[i]), run_time=0.3)
        cap.hold()

        q1 = s["points"][order[-1]]
        small = lab("too small:\nruns out of capacity", 21, INK2).move_to(
            P.p(1.15e7, 1.47), aligned_edge=LEFT
        )
        big = lab("too big:\nruns out of tokens", 21, INK2).next_to(
            P.p(q1["n"], q1["loss"]), LEFT, buff=0.3
        )
        self.play(
            FadeIn(small, shift=0.1 * DOWN),
            FadeOut(cur),
            *cap(
                "Too small, and a model runs out of capacity. Too big, and it runs out of tokens."
            ),
        )
        self.play(FadeIn(big, shift=0.1 * DOWN))
        cap.hold()

        para = parabola(P, 0)
        r = ring(P.p(s["n_opt"], s["l"]))
        nopt = lab(
            "N<sub>opt</sub> ≈ " + size(s["n_opt"]), 26, INK, "SEMIBOLD"
        ).next_to(r, DOWN, buff=0.12)
        self.play(
            Create(para),
            *cap(
                f"Loss against size is a U. Its trough is the best size here: about {size(s['n_opt'])}."
            ),
            run_time=1.4,
        )
        self.play(Create(r), FadeIn(nopt, shift=0.1 * UP))
        cap.hold(0.5)
        outro(self)


# ------------------------------------------------------------------ 2. why the trough moves


class S2(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Why the trough moves")
        P = lossplot()
        leg = legend()
        dots = [points(P, b) for b in range(3)]
        paras = [parabola(P, b) for b in range(3)]
        rings = [ring(P.p(s["n_opt"], s["l"])) for s in SW]
        self.play(
            FadeIn(k),
            FadeIn(P),
            FadeIn(leg[0]),
            FadeIn(dots[0]),
            FadeIn(paras[0]),
            FadeIn(rings[0]),
        )
        self.play(
            *cap("Spend more compute, and the whole U moves: down, and to the right.")
        )
        for b in (1, 2):
            self.play(
                FadeIn(leg[b]),
                LaggedStart(*[GrowFromCenter(d) for d in dots[b]], lag_ratio=0.1),
                run_time=0.9,
            )
            self.play(Create(paras[b]), Create(rings[b]), run_time=0.9)
        path = DashedVMobject(
            VMobject(stroke_color=INK, stroke_width=3).set_points_smoothly(
                [r.get_center() for r in rings]
            ),
            num_dashes=16,
        )
        self.play(Create(path))
        cap.hold()

        # the loss split into its two terms, from the fitted law
        Q = Plot((1e7, 4.2e8), (0, 0.3), 10.0, 4.7, P.ax.get_center(), xt=[(1e7, "10M"), (3e7, "30M"), (1e8, "100M"), (3e8, "300M")],
                 yt=[(y, f"{y:.1f}") for y in (0, 0.1, 0.2, 0.3)], xl="active parameters N", yl="loss above the floor")  # fmt: skip
        gone = VGroup(*dots, *paras, *rings, path, leg)
        self.play(FadeOut(gone), FadeOut(P.yt), FadeOut(P.yl), FadeOut(P.grid), FadeIn(Q.yt), FadeIn(Q.yl), FadeIn(Q.grid),
                  *cap("Why? The loss is a floor, plus a size term, plus a data term."))  # fmt: skip
        n = np.logspace(7.0, np.log10(4.2e8), 140)
        sz = LAW["A"] * (n / 1e7) ** -LAW["alpha"]
        lc = ValueTracker(np.log10(SW[0]["c"]))
        dat = lambda: (
            LAW["B"] * (10 ** lc.get_value() / (kap(n) * n) / 1e8) ** -LAW["beta"]
        )  # noqa: E731
        size_c = Q.line(n, sz, BLUE, 5)
        data_c = always_redraw(lambda: Q.line(n, dat(), ORANGE, 5))
        tot_c = always_redraw(lambda: Q.line(n, sz + dat(), INK, 6))

        def opt():
            y = sz + dat()
            i = int(np.argmin(y))
            return Q.p(n[i], y[i])

        tdot = always_redraw(lambda: dot(opt(), INK, 0.12))
        sl = (
            lab("size term  A·N<sup>−α</sup>", 24, BLUE, "SEMIBOLD")
            .move_to(Q.p(1.6e7, 0.03))
            .shift(0.4 * UP + 0.4 * RIGHT)
        )
        dl = lab("data term  B·D<sup>−β</sup>", 24, ORANGE, "SEMIBOLD").next_to(
            Q.p(1.6e8, 0.27), LEFT, buff=0.1
        )
        tl = lab("sum", 24, INK, "SEMIBOLD")
        tl.add_updater(lambda m: m.next_to(tdot, UP, buff=0.35))
        self.play(Create(size_c), FadeIn(sl), run_time=1.0)
        self.play(Create(data_c), FadeIn(dl), run_time=1.0)
        self.play(Create(tot_c), FadeIn(tdot), FadeIn(tl), run_time=1.0)
        cap.hold()
        self.play(
            *cap(
                "A bigger model shrinks the size term, but sees fewer tokens: the data term grows."
            )
        )
        cap.hold(0.5)

        cl = always_redraw(
            lambda: lab(
                "C = " + sci(10 ** lc.get_value()) + " FLOPs", 26, INK, "SEMIBOLD"
            ).move_to([3.4, 2.95, 0])
        )

        def ghost():
            return VGroup(
                tot_c.copy().clear_updaters().set_stroke(opacity=0.25),
                dot(opt(), INK, 0.08).set_opacity(0.35),
            )

        ghosts = VGroup(ghost())
        self.add(ghosts)
        self.play(
            FadeIn(cl),
            *cap(
                "Add compute: every size sees more tokens, so the best size moves right."
            ),
        )
        for b in (1, 2):
            self.play(
                lc.animate.set_value(np.log10(SW[b]["c"])),
                run_time=2.2,
                rate_func=linear,
            )
            ghosts.add(ghost())
        cap.hold()
        tl.clear_updaters()

        # the troughs on log axes
        R = Plot((4e17, 1e19), (2e7, 3e8), 7.0, 4.4, (0.5, 0.0), logy=True, xt=[(1e18, "10<sup>18</sup>"), (1e19, "10<sup>19</sup>")],
                 yt=[(3e7, "30M"), (1e8, "100M"), (3e8, "300M")], xl="compute C (FLOPs)", yl="best size N<sub>opt</sub>")  # fmt: skip
        for m in (data_c, tot_c, tdot, cl):
            m.clear_updaters()
        self.play(FadeOut(VGroup(Q, P.ax, P.xt, P.xl, size_c, data_c, tot_c, tdot, sl, dl, tl, ghosts, cl)), FadeIn(R),
                  *cap("Plot each budget's trough against its compute, both on log scales."))  # fmt: skip
        td = VGroup(
            *[dot(R.p(s["c"], s["n_opt"]), BUD[b], 0.12) for b, s in enumerate(SW)]
        )
        self.play(LaggedStart(*[GrowFromCenter(d) for d in td], lag_ratio=0.3))
        fit = np.polyfit(
            np.log([s["c"] for s in SW]), np.log([s["n_opt"] for s in SW]), 1
        )
        cs = np.array([5e17, 8e18])
        ln_ = R.line(cs, np.exp(np.polyval(fit, np.log(cs))), INK, 4)
        sl2 = lab("N<sub>opt</sub> ∝ C<sup>a</sup>", 32, INK, "SEMIBOLD").next_to(
            R.p(1.3e18, 1.2e8), UP + LEFT, buff=0.1
        )
        self.play(
            Create(ln_),
            Write(sl2),
            *cap(
                "They fall on a line: N<sub>opt</sub> grows as a power of compute, C<sup>a</sup>."
            ),
        )
        cap.hold()
        self.play(
            *cap("The exponent <i>a</i> is what a scaling law has to get right."),
            pulse(sl2),
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 3. the three approaches; approach 1


def mini_icon(i):
    if i == 0:
        g = VGroup(*[VMobject(stroke_color=mix(PALE, ORANGE, j / 3), stroke_width=4).set_points_smoothly(
            [[-1.0 + 0.3 * j, 0.7, 0], [-0.4 + 0.35 * j, -0.05 - 0.05 * j, 0], [1.0, -0.45 - 0.08 * j, 0]]) for j in range(4)])  # fmt: skip
        env = VMobject(stroke_color=INK, stroke_width=5).set_points_smoothly(
            [[-1.0, 0.7, 0], [-0.4, -0.05, 0], [0.3, -0.35, 0], [1.0, -0.69, 0]]
        )
        return VGroup(g, env)
    if i == 1:
        u = VMobject(stroke_color=ORANGE, stroke_width=5).set_points_smoothly(
            [[-1.0, 0.55, 0], [0, -0.4, 0], [1.0, 0.65, 0]]
        )
        return VGroup(u, ring(u.point_from_proportion(0.48), 0.16))
    return lab("E + A/N<sup>α</sup> + B/D<sup>β</sup>", 30, INK, "SEMIBOLD")


def cards():
    titles = ("Training-curve envelope", "IsoFLOP troughs", "Parametric fit")
    descs = ("Which size is ahead at each\ncompute, from training curves", "A parabola per budget,\na line through the troughs",
             "One formula for the loss,\nfit to every run at once")  # fmt: skip
    out = VGroup()
    for i, x in enumerate((-4.6, 0, 4.6)):
        box = RoundedRectangle(
            corner_radius=0.18,
            width=4.15,
            height=4.3,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_width=0,
        ).move_to([x, 0.15, 0])
        top = box.get_top()[1]
        num = text(str(i + 1), 44, ORANGE, weight="SEMIBOLD").move_to(
            [x, top - 0.55, 0]
        )
        ti = text(titles[i], 26, INK, weight="SEMIBOLD").move_to([x, top - 1.2, 0])
        ic = mini_icon(i).move_to([x, top - 2.25, 0])
        de = text(descs[i], 21, INK2, line_spacing=0.9).move_to([x, top - 3.45, 0])
        out.add(VGroup(box, num, ti, ic, de))
    return out


class S3(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Three approaches")
        cs = cards()
        self.play(FadeIn(k), LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in cs], lag_ratio=0.25), run_time=1.6,
                  *cap("Chinchilla (2022) estimated <i>a</i> three ways. We ran all three on Allie's sweep."))  # fmt: skip
        cap.hold(1.2)
        tb = tabs(0, k)
        self.play(*[Transform(cs[i][2], tb[0][i]) for i in range(3)], *[FadeOut(VGroup(cs[i][0], cs[i][1], cs[i][3], cs[i][4])) for i in range(3)],
                  FadeIn(tb[1]))  # fmt: skip
        self.remove(*[cs[i][2] for i in range(3)])
        self.add(tb)

        P = Plot((4e16, 8e18), (1.26, 1.62), 7.6, 4.4, (-2.55, 0.1), xt=[(1e17, "10<sup>17</sup>"), (1e18, "10<sup>18</sup>")],
                 yt=[(y, f"{y:.1f}") for y in (1.3, 1.4, 1.5, 1.6)], xl="compute spent so far (FLOPs)", yl="training loss")  # fmt: skip
        ns = sorted({cu["n"] for cu in A1["curves"]})
        col = {
            n: mix("#E3A57C", "#5E220A", i / (len(ns) - 1)) for i, n in enumerate(ns)
        }
        curves = VGroup(
            *[P.line(cu["c"], cu["y"], col[cu["n"]], 3) for cu in A1["curves"]]
        )
        self.play(
            FadeIn(P),
            *cap(
                "Approach 1 watches every run as it trains: loss against compute spent."
            ),
        )
        self.play(
            LaggedStart(*[Create(cv) for cv in curves], lag_ratio=0.04), run_time=3.0
        )
        cap.hold()
        env = P.line(A1["env_c"], A1["env_y"], INK, 7)
        self.play(
            Create(env, rate_func=linear),
            run_time=2.2,
            *cap(
                "At each compute keep the lowest curve: the <i>envelope</i>. Its size is ahead."
            ),
        )
        cap.hold()

        ec, en = np.array(A1["env_c"]), np.array(A1["env_n"])
        keep = ec >= 1e17
        ec, en = ec[keep], en[keep]
        sn, i = en.copy(), 0
        while i < len(en):
            j = i
            while j < len(en) and en[j] == en[i]:
                j += 1
            if j - i < 4 and i > 0 and j < len(en):
                sn[i:j] = sn[i - 1]
            i = j
        T = Plot((1e17, 8e18), (1.2e7, 4e8), 3.5, 1.75, (4.95, 1.25), logy=True, xt=[(1e17, "10<sup>17</sup>"), (1e18, "10<sup>18</sup>")],
                 yt=[(3e7, "30M"), (1e8, "100M")], yl="size on the envelope", px=19)  # fmt: skip
        stp = []
        for q in range(len(ec)):
            stp.append((ec[q], sn[q]))
            if q + 1 < len(ec) and sn[q + 1] != sn[q]:
                stp.append((ec[q + 1], sn[q]))
        steps = T.line([a for a, _ in stp], [b for _, b in stp], INK, 4)
        fit = np.polyfit(np.log(ec), np.log(en), 1)
        fl = T.line(
            ec[[0, -1]],
            np.exp(np.polyval(fit, np.log(ec[[0, -1]]))),
            ORANGE,
            3,
            dash=12,
        )
        al = lab(f"a = {A1['a']:.2f}", 26, ORANGE, "SEMIBOLD").next_to(
            T.p(1.2e17, 2.6e8), RIGHT, buff=0
        )
        self.play(
            FadeIn(T),
            Create(steps),
            run_time=1.4,
            *cap(
                f"That size grows as C<sup>{A1['a']:.2f}</sup>. But look at the envelope's level: a sawtooth."
            ),
        )
        self.play(Create(fl), FadeIn(al))
        jumps = VGroup(
            *[
                Circle(radius=0.32, stroke_color=ORANGE, stroke_width=5).move_to(
                    P.p(s["c"], 1.39)
                )
                for s in SW[:2]
            ]
        )
        self.play(LaggedStart(*[Create(j) for j in jumps], lag_ratio=0.4))
        cap.hold()

        L = Plot(
            (0, 1),
            (0, 1.15),
            3.5,
            1.45,
            (4.95, -1.4),
            logx=False,
            yl="learning rate",
            px=19,
        )
        fr, lr0 = SW[0]["c"] / SW[1]["c"], D["miss"]["lr_end_sweep"]
        long_ = L.line([0, 1], [1, lr0], INK2, 4)
        short = L.line([0, fr], [1, lr0], ORANGE, 5)
        mark = L.vline(fr, MUTED, 2, 0, 1.1)
        mdot = dot(L.p(fr, 1 - fr * (1 - lr0)), INK2, 0.08)
        ml = lab(f"still {1 - fr * (1 - lr0):.0%}", 19, INK2).next_to(
            mdot, UP + RIGHT, buff=0.08
        )
        ll = VGroup(lab("short run", 19, ORANGE, "SEMIBOLD").next_to(L.p(fr, 0.12), RIGHT, buff=0.1),
                    lab("long run", 19, INK2, "SEMIBOLD").next_to(L.p(0.8, 0.2), UP + RIGHT, buff=0.04))  # fmt: skip
        self.play(
            FadeIn(L),
            Create(long_),
            Create(short),
            FadeIn(ll),
            *cap(
                f"Each run decays its learning rate to {lr0:.0%} of its peak at its own end."
            ),
        )
        cap.hold()
        self.play(
            Create(mark),
            FadeIn(mdot),
            FadeIn(ml),
            *cap(
                "A run passing by mid-schedule hasn't decayed yet: its loss sits too high."
            ),
        )
        cap.hold()

        bi = A1["bias"]
        pair = [
            i
            for i, cu in enumerate(A1["curves"])
            if cu["shape"] == bi["shape"] and cu["budget"] in (bi["short"], bi["long"])
        ]
        a, b = P.p(bi["c"], bi["short_end"]), P.p(bi["c"], bi["long_mid"])
        arr = DoubleArrow(
            a,
            b,
            buff=0.05,
            stroke_width=4,
            color=ORANGE,
            tip_length=0.16,
            max_tip_length_to_length_ratio=0.2,
        )
        gl = (
            lab(f"{bi['gap']:.2f}", 26, ORANGE, "SEMIBOLD")
            .next_to(a, RIGHT, buff=0.2)
            .shift(0.3 * UP)
        )
        self.play(*[curves[i].animate.set_stroke(opacity=0.15) for i in range(len(curves)) if i not in pair], env.animate.set_stroke(opacity=0.25),
                  FadeOut(jumps), *[curves[i].animate.set_stroke(width=5) for i in pair],
                  *cap(f"Same size, same compute: the finished run is {bi['gap']:.2f} lower."))  # fmt: skip
        self.play(
            GrowFromCenter(arr),
            FadeIn(gl),
            FadeIn(dot(a, ORANGE, 0.09)),
            FadeIn(dot(b, col[A1["curves"][pair[0]]["n"]], 0.09)),
            FadeIn(
                lab("finished", 20, INK2).next_to(a, LEFT, buff=0.18).shift(0.12 * DOWN)
            ),
            FadeIn(lab("still decaying", 20, INK2).next_to(b, UP + RIGHT, buff=0.08)),
        )
        cap.hold()
        self.play(
            *cap("So the envelope gets the exponent right, but not the loss."),
            pulse(al),
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 4. approach 2


class S4(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Three approaches")
        tb = tabs(1, k)
        P = lossplot(center=(-3.05, 0.05), w=6.6, h=4.4, yr=(1.30, 1.46))
        dots = VGroup(*[points(P, b, hollow_out=True) for b in range(3)])
        self.play(
            FadeIn(k),
            FadeIn(tb),
            FadeIn(P),
            FadeIn(dots),
            *cap("Approach 2 goes budget by budget: a parabola in log N per slice,"),
        )
        cap.hold()
        paras = [parabola(P, b) for b in range(3)]
        self.play(
            LaggedStart(*[Create(p) for p in paras], lag_ratio=0.3),
            run_time=1.8,
            *cap("fit through the four sizes nearest the best (filled dots)."),
        )
        cap.hold()
        tr = [trough(P, b, DOWN if b == 1 else UP) for b in range(3)]
        self.play(*cap("Each vertex is a trough: " + ", ".join(size(s["n_opt"]) for s in SW) + " active parameters."),
                  LaggedStart(*[Create(r) for r, _ in tr], lag_ratio=0.3), LaggedStart(*[FadeIn(t) for _, t in tr], lag_ratio=0.3))  # fmt: skip
        cap.hold()

        R = Plot((4e17, 1e19), (2e7, 3e8), 5.4, 4.2, (3.85, 0.05), logy=True, xt=[(1e18, "10<sup>18</sup>"), (1e19, "10<sup>19</sup>")],
                 yt=[(3e7, "30M"), (1e8, "100M"), (3e8, "300M")], xl="compute C (FLOPs)", yl="trough N<sub>opt</sub>")  # fmt: skip
        td = VGroup(
            *[dot(R.p(s["c"], s["n_opt"]), BUD[b], 0.12) for b, s in enumerate(SW)]
        )
        lc = np.log([s["c"] for s in SW])
        fit = np.polyfit(lc, np.log([s["n_opt"] for s in SW]), 1)
        cs = np.array([5e17, 8e18])
        fl = R.line(cs, np.exp(np.polyval(fit, np.log(cs))), INK, 4)
        al = lab(f"a = {A2['a']:.2f}", 30, INK, "SEMIBOLD").next_to(
            R.p(6e17, 1.6e8), RIGHT, buff=0
        )
        self.play(FadeIn(R), *[TransformFromCopy(tr[b][0], td[b]) for b in range(3)],
                  *cap(f"A straight line through the three troughs: N<sub>opt</sub> ∝ C<sup>{A2['a']:.2f}</sup>."))  # fmt: skip
        self.play(Create(fl), Write(al))
        cap.hold()

        bars = VGroup(
            *[
                Line(R.p(s["c"], lo), R.p(s["c"], hi), color=BUD[b], stroke_width=5)
                for b, (s, (lo, hi)) in enumerate(zip(SW, A2["nopt_ci"]))
            ]
        )
        mx, my = lc.mean(), np.log([s["n_opt"] for s in SW]).mean()
        fan = VGroup(
            *[
                R.line(cs, np.exp(my + a * (np.log(cs) - mx)), MUTED, 2.5, dash=14)
                for a in A2["ci"]
            ]
        )
        lo, hi = A2["ci"]
        self.play(
            *[GrowFromCenter(b_) for b_ in bars],
            *cap(
                f"But three points make a shaky line: 90% interval {lo:.2f} to {hi:.2f}."
            ),
        )
        self.play(Create(fan), run_time=1.2)
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 5. approach 3


class S5(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Three approaches")
        tb = tabs(2, k)
        parts = [
            "L(N, D)  =",
            "E",
            "+",
            "A·(N/10<sup>7</sup>)<sup>−α</sup>",
            "+",
            "B·(D/10<sup>8</sup>)<sup>−β</sup>",
        ]
        f = VGroup(*[lab(s, 34, INK, "SEMIBOLD") for s in parts]).arrange(
            RIGHT, buff=0.22
        )
        base = f[0][0].get_bottom()[1]
        for i in (1, 3, 5):
            f[i].shift((base - f[i][0].get_bottom()[1]) * UP)
        for i in (2, 4):
            f[i].set_y(f[1].get_center()[1])
        f.move_to([0, 2.45, 0])
        self.play(
            FadeIn(k),
            FadeIn(tb),
            Write(f),
            run_time=1.5,
            *cap(
                f"Approach 3 fits one formula to all {M['moe_runs']} MoE runs at once."
            ),
        )
        names = (("floor", 1, INK2), ("size term", 3, BLUE), ("data term", 5, ORANGE))
        under = VGroup(
            *[
                lab(s, 22, c, "SEMIBOLD")
                .next_to(f[i], DOWN, buff=0.2)
                .set_y(f.get_bottom()[1] - 0.3)
                for s, i, c in names
            ]
        )
        self.play(*[f[i].animate.set_color(c) for _, i, c in names], LaggedStart(*[FadeIn(u, shift=0.1 * UP) for u in under], lag_ratio=0.3),
                  *cap("A floor E, plus a size term and a data term, each a power law."))  # fmt: skip
        cap.hold()

        P = lossplot(center=(-1.55, -0.42), w=8.6, h=3.6, yr=(1.26, 1.50))
        dots = VGroup(*[points(P, b) for b in range(3)])
        th = ValueTracker(0.0)
        p0 = dict(E=1.20, A=0.05, alpha=0.3, B=0.6, beta=0.4)

        def par():
            t = th.get_value()
            return {k_: p0[k_] * (LAW[k_] / p0[k_]) ** t for k_ in p0}

        curves = always_redraw(
            lambda: VGroup(
                *[isocurve(P, s["c"], par(), BUD[b]) for b, s in enumerate(SW)]
            )
        )
        self.play(
            FadeIn(P),
            FadeIn(dots),
            *cap(
                "Fit it to every run together: its cut along each budget passes through the runs."
            ),
        )
        self.play(FadeIn(curves))
        self.play(th.animate.set_value(1.0), run_time=2.5, rate_func=smooth)
        curves.clear_updaters()
        cap.hold()

        vals = (
            f"E = {LAW['E']:.3f}",
            f"α = {LAW['alpha']:.2f}",
            f"β = {LAW['beta']:.2f}",
        )
        nv = VGroup(
            *[
                lab(v, 24, c, "SEMIBOLD").move_to(u)
                for v, u, (_, _, c) in zip(vals, under, names)
            ]
        )
        floor = P.hline(LAW["E"], INK2, 2.5)
        fl = lab(f"floor E = {LAW['E']:.3f}", 21, INK2).next_to(
            P.p(1.1e7, LAW["E"]), UP + RIGHT, buff=0.08
        )
        self.play(*[Transform(u, v) for u, v in zip(under, nv)], Create(floor), FadeIn(fl),
                  *cap(f"The fit: floor E = {LAW['E']:.3f}, α = {LAW['alpha']:.2f}, β = {LAW['beta']:.2f}."))  # fmt: skip
        cap.hold()
        rm = VGroup(lab("residuals", 22, INK2), lab(f"{LAW['rmse'] * 1e3:.1f}×10<sup>−3</sup> nats", 30, INK, "SEMIBOLD"),
                    lab(f"seed noise {M['sigma'] * 1e3:.1f}×10<sup>−3</sup>", 22, INK2)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)  # fmt: skip
        rm.move_to([4.25, 0.45, 0], aligned_edge=LEFT)
        self.play(
            FadeIn(rm, shift=0.1 * UP),
            *cap("It fits the runs to within their seed-to-seed noise."),
        )
        cap.hold()
        lo, hi = LAW["ci"]["a"]
        av = VGroup(lab("a = β / (α + β)", 26, INK, "SEMIBOLD"), lab(f"= {LAW['a']:.2f}", 40, ORANGE, "SEMIBOLD"),
                    lab(f"90%: {lo:.2f} – {hi:.2f}", 22, INK2)).arrange(DOWN, buff=0.14, aligned_edge=LEFT)  # fmt: skip
        av.next_to(rm, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(
            FadeIn(av, shift=0.1 * UP),
            *cap(
                f"The exponents give the best size: a = β/(α+β) = {LAW['a']:.2f}, the tightest estimate."
            ),
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 6. the exponents agree


class S6(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("The exponents agree")
        E = D["exponents"]
        rows = (("1  Envelope", E["moe"]["envelope"], None), ("2  Troughs", E["moe"]["troughs"], E["moe"]["troughs_ci"]),
                ("3  Parametric", E["moe"]["law"], E["moe"]["law_ci"]))  # fmt: skip
        N = Plot((0.3, 0.9), (0, 4), 10.0, 3.3, (1.0, 0.8), logx=False, xt=[(v, f"{v:.1f}") for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
                 xl="a  in  N<sub>opt</sub> ∝ C<sup>a</sup>")  # fmt: skip
        ys = (3.5, 2.6, 1.7, 0.6)
        names = VGroup(
            *[
                text(n, 26, INK, weight="SEMIBOLD").next_to(
                    N.p(0.3, y), LEFT, buff=0.35
                )
                for (n, _, _), y in zip(rows, ys)
            ]
        )
        self.play(
            FadeIn(k),
            FadeIn(N.ax),
            FadeIn(N.xt),
            FadeIn(N.xl),
            FadeIn(names),
            *cap("Three methods on one sweep. Put their exponents side by side."),
        )
        marks = VGroup()
        for (_, a, ci), y in zip(rows, ys):
            g = VGroup()
            if ci:
                g.add(Line(N.p(ci[0], y), N.p(ci[1], y), color=PALE, stroke_width=10))
            g.add(
                dot(N.p(a, y), ORANGE, 0.14),
                lab(f"{a:.2f}", 26, ORANGE, "SEMIBOLD").next_to(
                    N.p(a, y), UP, buff=0.2
                ),
            )
            marks.add(g)
        self.play(
            LaggedStart(*[FadeIn(m, shift=0.2 * DOWN) for m in marks], lag_ratio=0.35),
            run_time=1.8,
        )
        cap.hold()
        lo, hi = min(r[1] for r in rows), max(r[1] for r in rows)
        x0, x1 = N.p(lo, 0)[0], N.p(hi, 0)[0]
        y0, y1 = N.p(0, 1.25)[1], N.p(0, 3.95)[1]
        band = Rectangle(
            width=x1 - x0,
            height=y1 - y0,
            fill_color=ORANGE,
            fill_opacity=0.1,
            stroke_width=0,
        ).move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
        self.add(band)
        self.bring_to_back(band)
        self.play(FadeIn(band), *cap(f"They agree: a ≈ {lo:.2f} to {hi:.2f}."))
        cap.hold()
        cn = text("Chinchilla, language", 26, MUTED, weight="SEMIBOLD").next_to(
            N.p(0.3, ys[3]), LEFT, buff=0.35
        )
        chd = VGroup(*[dot(N.p(v, ys[3]), GREY, 0.1) for v in E["chinchilla"]])
        ch = N.vline(0.5, GREY, 2.5, 0.2, 3.95)
        self.play(FadeIn(cn), LaggedStart(*[GrowFromCenter(d) for d in chd], lag_ratio=0.2), Create(ch),
                  *cap("Chinchilla found about 0.5 for language: grow parameters and tokens equally."))  # fmt: skip
        cap.hold()
        a = E["moe"]["law"]
        u = 1.05
        head = lab("at 10× compute", 24, INK2).move_to([-5.3, -2.3, 0])
        rowsb = VGroup()
        for i, (nm, m, c) in enumerate(
            (("parameters", 10**a, ORANGE), ("tokens", 10 ** (1 - a), BLUE))
        ):
            y = -1.98 - 0.6 * i
            nl = lab(nm, 24, c, "SEMIBOLD").move_to([-2.65, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(
                width=u * m, height=0.3, fill_color=c, fill_opacity=0.85, stroke_width=0
            ).move_to([-2.4, y, 0], aligned_edge=LEFT)
            vl = lab(f"×{m:.1f}", 26, c, "SEMIBOLD").next_to(bar, RIGHT, buff=0.15)
            rowsb.add(VGroup(nl, bar, vl))
        eq = DashedLine(
            [-2.4 + u * 10**0.5, -1.75, 0],
            [-2.4 + u * 10**0.5, -2.8, 0],
            color=GREY,
            stroke_width=2.5,
            dash_length=0.08,
        )
        eql = lab(f"equal split ×{10**0.5:.1f}", 20, GREY).next_to(eq, DOWN, buff=0.05)
        self.play(FadeIn(head), *[GrowFromEdge(r[1], LEFT) for r in rowsb], *[FadeIn(VGroup(r[0], r[2])) for r in rowsb], Create(eq), FadeIn(eql),
                  *cap("For Allie, the model should grow faster than its data."))  # fmt: skip
        cap.hold(0.8)
        outro(self)


# ------------------------------------------------------------------ 7. the Allie 2.0 miss


def frontier(center=(-0.35, 0.1), w=11.2, h=4.6):
    return Plot((4e17, 3e21), (1.18, 1.39), w, h, center, xt=[(10.0**e, f"10<sup>{e}</sup>") for e in (18, 19, 20, 21)],
                yt=[(y, f"{y:.2f}") for y in (1.20, 1.25, 1.30, 1.35)], xl="training compute (FLOPs)", yl="loss (nats)")  # fmt: skip


def trough_dots(P):
    return VGroup(*[dot(P.p(s["c"], s["l"]), BUD[b], 0.11) for b, s in enumerate(SW)])


def law_frontier(P, width=5, cmax=2.2e21):
    c = np.array(LAW["lstar_c"])
    return P.line(c[c <= cmax], np.array(LAW["lstar"])[c <= cmax], ORANGE, width)


def nofloor(P, width=4, cmax=2.2e21):
    cs = np.logspace(np.log10(4.5e17), np.log10(cmax), 40)
    return P.line(
        cs, np.exp(np.polyval(A2["loglaw"], np.log(cs))), BLUE, width, dash=34
    )


class S7(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("The Allie 2.0 test")
        P = frontier()
        td = trough_dots(P)
        sw = (
            lab("sweep troughs", 22, INK2).next_to(td, DOWN, buff=0.2).shift(0.2 * LEFT)
        )
        self.play(FadeIn(k), FadeIn(P), LaggedStart(*[GrowFromCenter(d) for d in td], lag_ratio=0.2), FadeIn(sw),
                  *cap("The sweep's three troughs, on a compute axis that reaches the big runs."))  # fmt: skip
        cap.hold()
        c20 = M["C20"]
        x20 = P.p(c20, 1.3)[0]
        v20 = P.vline(c20, RULE, 2.5)
        l20 = lab(f"Allie 2.0\n{sci(c20)} FLOPs", 22, INK, "SEMIBOLD").next_to(
            P.p(c20, 1.39), DOWN + RIGHT, buff=0.1
        )
        arr = Line(
            P.p(SW[2]["c"] * 1.25, 1.372),
            P.p(c20 / 1.25, 1.372),
            color=INK2,
            stroke_width=3,
        ).add_tip(tip_length=0.15)
        xl = lab(f"{c20 / SW[2]['c']:.0f}× past the sweep", 22, INK2).next_to(
            arr, UP, buff=0.08
        )
        self.play(
            Create(v20),
            FadeIn(l20),
            GrowFromEdge(arr, LEFT),
            FadeIn(xl),
            *cap(
                f"Allie 2.0 used {c20 / SW[2]['c']:.0f} times the sweep's biggest budget."
            ),
        )
        cap.hold()

        lf = law_frontier(P)
        fl = P.hline(LAW["E"], ORANGE, 2)
        fll = lab(f"floor {LAW['E']:.3f}", 20, ORANGE).next_to(
            P.p(5e17, LAW["E"]), DOWN, buff=0.08, aligned_edge=LEFT
        )
        ld = dot(P.p(c20, LAW["f20"]), ORANGE, 0.12)
        ll = lab(f"law: {LAW['f20']:.3f}", 24, ORANGE, "SEMIBOLD").next_to(
            ld, UP + RIGHT, buff=0.06
        )
        self.play(
            Create(lf),
            run_time=1.6,
            *cap(
                f"Approach 3's law flattens toward its floor. For Allie 2.0: {LAW['f20']:.3f}."
            ),
        )
        self.play(Create(fl), FadeIn(fll), GrowFromCenter(ld), FadeIn(ll))
        cap.hold()
        nf = nofloor(P)
        bd = dot(P.p(c20, A2["pow20"]), BLUE, 0.12)
        bl = lab(f"troughs: {A2['pow20']:.3f}", 24, BLUE, "SEMIBOLD").next_to(
            bd, DOWN + LEFT, buff=0.06
        )
        self.play(
            Create(nf),
            run_time=1.6,
            *cap(f"Approach 2's troughs, extended as a power law: {A2['pow20']:.3f}."),
        )
        self.play(GrowFromCenter(bd), FadeIn(bl))
        cap.hold()

        st = star(P.p(c20, 1.345))
        self.play(
            FadeIn(st),
            *cap(
                f"Allie 2.0's big run, before its second anneal, scored {M['Y20']:.3f}."
            ),
        )
        self.play(
            st.animate.move_to(P.p(c20, M["Y20"])), run_time=1.2, rate_func=smooth
        )
        sl = (
            lab(f"actual: {M['Y20']:.3f}", 26, INK, "SEMIBOLD")
            .next_to(st, LEFT, buff=0.2)
            .shift(0.22 * UP)
        )
        self.play(FadeIn(sl))
        cap.hold()
        y20 = P.p(c20, M["Y20"])[1]
        xa = x20 + 0.4
        up = DoubleArrow(
            [xa, y20 + 0.05, 0],
            [xa, ld.get_center()[1], 0],
            buff=0,
            color=ORANGE,
            stroke_width=4,
            tip_length=0.13,
        )
        dn = DoubleArrow(
            [xa, y20 - 0.05, 0],
            [xa, bd.get_center()[1], 0],
            buff=0,
            color=BLUE,
            stroke_width=4,
            tip_length=0.13,
        )
        ul = (
            lab(f"+{LAW['f20'] - M['Y20']:.3f}", 22, ORANGE, "SEMIBOLD")
            .next_to(up, RIGHT, buff=0.1)
            .shift(0.16 * DOWN)
        )
        dl = lab(f"−{M['Y20'] - A2['pow20']:.3f}", 22, BLUE, "SEMIBOLD").next_to(
            dn, RIGHT, buff=0.1
        )
        self.play(
            GrowFromCenter(up),
            GrowFromCenter(dn),
            FadeIn(ul),
            FadeIn(dl),
            *cap(
                "The law missed high, the troughs missed low, by about the same amount."
            ),
        )
        cap.hold()
        lo, hi = LAW["ci"]["f20"]
        half = (hi - lo) / 2
        xc = x20 - 0.24
        ya, yb = P.p(c20, lo)[1], P.p(c20, hi)[1]
        ci = VGroup(
            Line([xc, ya, 0], [xc, yb, 0]),
            Line([xc - 0.08, ya, 0], [xc + 0.08, ya, 0]),
            Line([xc - 0.08, yb, 0], [xc + 0.08, yb, 0]),
        )
        ci.set_stroke(ORANGE, 4)
        self.play(
            GrowFromCenter(ci),
            *cap(
                f"The law's miss is {(LAW['f20'] - M['Y20']) / half:.0f}× its own error bar: not noise, something systematic."
            ),
        )
        cap.hold()
        self.play(
            *cap(
                "Same exponents, opposite errors: the sweep pins the shape, not the level."
            )
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 8. the data term


class S8(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("The data term")
        B = D["beta"]
        prof = B["profile"]
        bv = np.array([r["beta"] for r in prof])
        cb = D["exponents"]["chinchilla_beta"]
        f = lab(
            f"L = E + A·N<sup>−α</sup> + <span foreground='{ORANGE}'>B·D<sup>−β</sup></span>",
            34,
            INK,
            "SEMIBOLD",
        ).move_to([-3.4, 2.5, 0])
        lo, hi = B["lo"], B["hi"]
        self.play(
            FadeIn(k),
            Write(f),
            *cap(
                f"Could another data exponent β fix it? Chinchilla's is {cb:.2f}; ours is about {B['free_ls']:.1f}."
            ),
        )
        cap.hold()

        P = lossplot(center=(-3.0, -0.4), w=6.1, h=3.6, yr=(1.22, 1.50), px=20)
        dots = VGroup(*[points(P, b, r=0.075) for b in range(3)])
        bt = ValueTracker(B["free_ls"])

        def par():
            b = bt.get_value()
            return dict(
                beta=b,
                **{
                    k_: float(np.interp(b, bv, [r[k_] for r in prof]))
                    for k_ in ("E", "A", "alpha", "B")
                },
            )

        def draw():
            p = par()
            g = VGroup()
            for b, s in enumerate(SW):
                g.add(isocurve(P, s["c"], p, BUD[b]))
                for q in s["points"]:
                    y = iso(q["n"], s["c"], p)
                    if abs(y - q["loss"]) > 0.002 and P.inside(q["loss"]):
                        g.add(
                            Line(
                                P.p(q["n"], q["loss"]),
                                P.p(q["n"], np.clip(y, 1.22, 1.5)),
                                color=BUD[b],
                                stroke_width=2.5,
                            )
                        )
            return g

        curves = always_redraw(draw)
        Q = Plot((0.15, 1.6), (0, np.log10(1500)), 5.4, 3.6, (3.75, -0.4), logx=False,
                 xt=[(v, f"{v:.2f}".rstrip("0").rstrip(".")) for v in (0.28, 0.5, 1.0, 1.5)],
                 yt=[(np.log10(1 + v), s) for v, s in ((0, "0"), (10, "10"), (100, "100"), (1000, "1000"))],
                 xl="β, held fixed", yl="misfit to the runs, Δχ²", px=20)  # fmt: skip
        dchi = np.array([max(r["dchi2"], 0) for r in prof])
        prof_c = Q.line(bv, np.log10(1 + dchi), INK, 4)
        okb = Rectangle(
            width=Q.p(hi, 0)[0] - Q.p(lo, 0)[0],
            height=Q.ax.height,
            fill_color=ORANGE,
            fill_opacity=0.15,
            stroke_width=0,
        )
        okb.move_to([(Q.p(lo, 0)[0] + Q.p(hi, 0)[0]) / 2, Q.ax.get_center()[1], 0])
        dq = lambda: max(float(np.interp(bt.get_value(), bv, dchi)), 0)  # noqa: E731
        mdot = always_redraw(
            lambda: dot(Q.p(bt.get_value(), np.log10(1 + dq())), ORANGE, 0.12)
        )
        read = always_redraw(lambda: VGroup(lab(f"β = {bt.get_value():.2f}", 26, ORANGE, "SEMIBOLD"), lab(f"Δχ² = {dq():.0f}", 26, INK, "SEMIBOLD"))
                             .arrange(RIGHT, buff=0.45).move_to([3.75, 2.5, 0]))  # fmt: skip
        self.play(FadeIn(P), FadeIn(dots), FadeIn(curves), FadeIn(Q), FadeIn(okb), Create(prof_c), FadeIn(mdot), FadeIn(read),
                  *cap(f"Nothing holds β in place: no bound binds in {B['bootstrap_draws']:,} bootstrap refits."))  # fmt: skip
        cap.hold()
        self.play(
            *cap(
                f"Profile it: the runs alone pin β between {lo:.2f} and {hi:.2f} (90%)."
            ),
            pulse(okb),
        )
        cap.hold()
        self.play(
            bt.animate.set_value(0.5),
            run_time=3.0,
            *cap(
                f"Force β down to 0.5 and refit the rest: the curves miss, Δχ² ≈ {B['dchi2_050']:.0f}."
            ),
        )
        cap.hold()
        self.play(
            bt.animate.set_value(cb),
            run_time=1.5,
            *cap(
                f"At Chinchilla's {cb:.2f}, Δχ² ≈ {B['dchi2_028']:.0f}. The sweep rules it out."
            ),
        )
        cap.hold()
        self.play(bt.animate.set_value(B["hit20"]), run_time=1.6,
                  *cap(f"Even the β that would hit Allie 2.0, {B['hit20']:.2f}, costs Δχ² ≈ {B['hit20_dchi2']:.0f}."))  # fmt: skip
        cap.hold()
        fl = D["floor"]
        yb0, yb1 = P.p(1e8, fl["lo"])[1], P.p(1e8, fl["hi"])[1]
        band = Rectangle(
            width=P.ax.width,
            height=yb1 - yb0,
            fill_color=INK2,
            fill_opacity=0.16,
            stroke_width=0,
        ).move_to([P.ax.get_center()[0], (yb0 + yb1) / 2, 0])
        a20 = P.hline(M["Y20"], INK, 3)
        x0 = P.p(4.2e8, 1.3)[0]
        bl = (
            lab(f"floor the sweep allows: {fl['lo']:.3f}–{fl['hi']:.3f}", 20, INK2)
            .next_to(band, UP, buff=0.06)
            .align_to([x0, 0, 0], RIGHT)
        )
        al = (
            lab(f"Allie 2.0's big run: {M['Y20']:.3f}", 20, INK, "SEMIBOLD")
            .next_to(a20, DOWN, buff=0.06)
            .align_to([x0, 0, 0], RIGHT)
        )
        self.play(bt.animate.set_value(B["free_ls"]), FadeIn(band), FadeIn(bl), Create(a20), FadeIn(al), run_time=1.6,
                  *cap("Within this law the floor is pinned too, and every allowed floor sits above Allie 2.0."))  # fmt: skip
        cap.hold()
        self.play(
            *cap("No exponent or floor in this law fits both the sweep and Allie 2.0.")
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 8b. why the law missed Allie 2.0

W = D["miss"]


def signed(v, d=3):
    return f"{v:+.{d}f}".replace("-", "−")


def how(r):
    return f"{'2' if '+' in r['fit'] else '3'} budgets → {'Allie 2.0' if 'Allie' in r['target'] else 'the 3rd'}"


class M1(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Why the law missed")
        bt = W["backtest"]
        P = Plot((1.0, 300), (-0.08, 0.046), 10.4, 4.0, (0.7, 0.45),
                 xt=[(r["dist"], f"{r['dist']:.1f}×".replace(".0×", "×") if r["dist"] < 10 else f"{r['dist']:.0f}×") for r in bt["law"]],
                 yt=[(y, "0" if y == 0 else signed(y, 2)) for y in (0.04, 0.02, 0, -0.02, -0.04, -0.06)],
                 xl="how far out: target compute ÷ largest compute fitted", yl="forecast − actual (nats)")  # fmt: skip
        zero = Line(P.p(1.0, 0), P.p(300, 0), color=INK2, stroke_width=2)
        hows = VGroup(
            *[
                lab(how(r), 19, MUTED).next_to(t, DOWN, buff=0.06)
                for r, t in zip(bt["law"], P.xt)
            ]
        )
        P.xl.next_to(hows, DOWN, buff=0.12).set_x(P.ax.get_center()[0])
        self.play(
            FadeIn(k),
            FadeIn(P),
            Create(zero),
            *cap("Why did the law miss? First: could a backtest have warned us?"),
        )
        cap.hold()

        series = (("law", bt["law"], ORANGE), ("troughs", bt["troughs"], BLUE))
        first = VGroup()
        for name, rows, col in series:
            r = rows[0]
            d0 = dot(P.p(r["dist"], r["err"]), col, 0.11)
            t0 = lab(
                f"{name}  {signed(r['err'], 4 if abs(r['err']) < 0.001 else 3)}",
                22,
                col,
                "SEMIBOLD",
            ).next_to(d0, (UP if name == "law" else DOWN) + LEFT, buff=0.06)
            first.add(VGroup(d0, t0))
        self.play(FadeIn(hows[0]), LaggedStart(*[FadeIn(g, scale=0.8) for g in first], lag_ratio=0.3),
                  *cap(f"Fit two budgets and predict the third, {bt['law'][0]['dist']:.1f}× out: the law is nearly exact."))  # fmt: skip
        cap.hold()
        lines, ends = VGroup(), VGroup()
        for name, rows, col in series:
            lines.add(
                P.line([r["dist"] for r in rows], [r["err"] for r in rows], col, 4)
            )
            for r in rows[1:]:
                d = dot(P.p(r["dist"], r["err"]), col, 0.11)
                ends.add(
                    VGroup(
                        d,
                        lab(signed(r["err"]), 22, col, "SEMIBOLD").next_to(
                            d, UP if r["err"] > 0 else DOWN, buff=0.12
                        ),
                    )
                )
        self.bring_to_back(lines)
        self.play(Create(lines), FadeIn(hows[1:]), LaggedStart(*[FadeIn(e) for e in ends], lag_ratio=0.15), run_time=2.0,
                  *cap(f"At Allie 2.0, {bt['law'][1]['dist']:.0f}× out, the law is {bt['law'][1]['err']:.3f} too high. Its backtest gave no warning."))  # fmt: skip
        cap.hold()

        R = W["ruled_out"]
        chips = VGroup()
        for title, body in (("Same eval", f"identical positions,\n{R['eval_moves']:,} moves"),
                            ("Parameter counting", f"moves the forecast\nby under {R['nparams_max']:.4f}"),
                            ("Training recipe", f"2.0's recipe is {R['recipe_cost'][0]:.3f}–{R['recipe_cost'][1]:.3f}\nworse at sweep scale:\nit widens the gap")):  # fmt: skip
            g = VGroup(text("RULED OUT", 20, MUTED, weight="SEMIBOLD"), text(title, 32, INK, weight="SEMIBOLD"),
                       text(body, 25, INK2, line_spacing=0.9)).arrange(DOWN, buff=0.22)  # fmt: skip
            box = RoundedRectangle(
                corner_radius=0.18,
                width=4.0,
                height=3.2,
                fill_color=PANEL,
                fill_opacity=1,
                stroke_width=0,
            )
            chips.add(VGroup(box, g.move_to(box)))
        chips.arrange(RIGHT, buff=0.4).move_to([0, 0.2, 0])
        self.play(
            FadeOut(VGroup(P, zero, hows, first, lines, ends)),
            *cap("Not the cause: the eval, how parameters are counted, or the recipe."),
        )
        self.play(
            LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in chips], lag_ratio=0.3),
            run_time=1.2,
        )
        cap.hold(0.8)
        outro(self)


class M2(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Why the law missed")
        w, y20 = W["waterfall"], M["Y20"]
        P = Plot((0, 3.6), (1.246, 1.292), 5.3, 4.2, (-3.35, 0.3), logx=False,
                 yt=[(y, f"{y:.2f}") for y in (1.25, 1.26, 1.27, 1.28, 1.29)], yl="loss (nats)")  # fmt: skip
        top = DashedLine(
            P.p(0, w[0]),
            P.p(3.6, w[0]),
            color=ORANGE,
            stroke_width=2.5,
            dash_length=0.1,
        )
        bot = DashedLine(
            P.p(0, y20), P.p(3.6, y20), color=INK, stroke_width=2.5, dash_length=0.1
        )
        tl = lab(f"law {w[0]:.3f}", 21, ORANGE, "SEMIBOLD").next_to(
            P.p(0.05, w[0]), UP, buff=0.06, aligned_edge=LEFT
        )
        bl = lab(f"Allie 2.0 {y20:.3f}", 21, INK, "SEMIBOLD").next_to(
            P.p(0.05, y20), DOWN, buff=0.06, aligned_edge=LEFT
        )
        y0, y1 = P.p(0, W["forms_lo"])[1], P.p(0, W["forms_hi"])[1]
        forms = Rectangle(
            width=P.ax.width,
            height=y1 - y0,
            fill_color=ORANGE,
            fill_opacity=0.13,
            stroke_width=0,
        )
        forms.move_to([P.ax.get_center()[0], (y0 + y1) / 2, 0])
        fl = (
            lab("every law form that fits the sweep", 20, ORANGE)
            .move_to(forms.get_corner(DOWN + RIGHT), aligned_edge=DOWN + RIGHT)
            .shift(0.08 * UP + 0.1 * LEFT)
        )
        self.add(forms)
        self.play(FadeIn(k), FadeIn(P), Create(top), Create(bot), FadeIn(tl), FadeIn(bl), FadeIn(forms), FadeIn(fl),
                  *cap(f"Every law form that fits all {M['moe_runs']} runs says {W['forms_lo']:.3f}–{W['forms_hi']:.3f}. Allie 2.0 scored {y20:.3f}."))  # fmt: skip
        cap.hold()

        Z = W["decay"]
        Q = Plot((800, 2.2e5), (-0.016, 0.0015), 4.7, 3.6, (3.75, 0.1), xt=[(1e3, "1K"), (1e4, "10K"), (1e5, "100K")],
                 yt=[(v, "0" if v == 0 else signed(v)) for v in (0, -0.005, -0.01)], xl="training steps",
                 yl=f"loss change from ending at {W['lr_end_20']:.1%}, not {W['lr_end_sweep']:.0%}", px=20)  # fmt: skip
        sx0, sx1 = Q.p(Z["sweep_steps"][0], 0)[0], Q.p(Z["sweep_steps"][1], 0)[0]
        sw = Rectangle(
            width=sx1 - sx0,
            height=Q.ax.height,
            fill_color=BUD[0],
            fill_opacity=0.16,
            stroke_width=0,
        )
        sw.move_to([(sx0 + sx1) / 2, Q.ax.get_center()[1], 0])
        swl = lab("sweep runs", 20, BUD[2], "SEMIBOLD").next_to(sw, DOWN, buff=-0.45)
        qz = Line(Q.p(800, 0), Q.p(2.2e5, 0), color=INK2, stroke_width=2)
        self.play(FadeOut(forms), FadeOut(fl), FadeIn(Q), FadeIn(qz), FadeIn(sw), FadeIn(swl),
                  *cap(f"The sweep runs decay the learning rate to {W['lr_end_sweep']:.0%} of peak; Allie 2.0 went to {W['lr_end_20']:.1%}."))  # fmt: skip
        cap.hold()
        st = np.logspace(np.log10(Z["s0"]), np.log10(W["steps20"]), 60)
        curve = Q.line(st, Z["k"] * np.log(st / Z["s0"]), INK, 4)
        meas = VGroup(
            *[dot(Q.p(s, g), ORANGE, 0.1) for s, g in zip(Z["steps"], Z["gain"])]
        )
        ml = lab(f"Allie 2.0: {signed(Z['gain'][-1])}", 20, ORANGE, "SEMIBOLD").next_to(
            meas[-1], DOWN + LEFT, buff=0.1
        )
        b1 = self.bar(P, 1, w[0], w[1], INK2)
        self.play(
            Create(curve),
            FadeIn(meas),
            FadeIn(ml),
            *cap(
                f"That last stretch pays more the longer the run: {signed(w[1] - w[0])} at Allie 2.0's length."
            ),
        )
        self.play(GrowFromEdge(b1[0], UP), FadeIn(b1[1:]))
        cap.hold()
        b2 = self.bar(P, 2, w[1], w[2], ORANGE, tentative=True)
        self.play(
            GrowFromEdge(b2[0], UP),
            FadeIn(b2[1:]),
            *cap(
                f"Tentatively {signed(w[2] - w[1])}: token-starved runs on the U's steep side set the law's floor."
            ),
        )
        cap.hold()
        lo, hi = W["rest_range"]
        b3 = self.bar(P, 3, w[2], w[3], GREY, sub=f"({signed(-lo)} to {signed(-hi)})")
        self.play(
            GrowFromEdge(b3[0], UP),
            FadeIn(b3[1:]),
            *cap(
                f"About {w[2] - w[3]:.3f} is left that nothing the sweep measured explains."
            ),
        )
        cap.hold(0.6)
        outro(self)

    @staticmethod
    def bar(P, x, a, b, color, tentative=False, sub=""):
        names = {1: f"decay to {W['lr_end_20']:.1%}", 2: "floor", 3: "unexplained"}
        x0, x1 = P.p(x - 0.33, a)[0], P.p(x + 0.33, a)[0]
        ya, yb = P.p(0, a)[1], P.p(0, b)[1]
        r = Rectangle(width=x1 - x0, height=ya - yb, fill_color=color, fill_opacity=0.35 if tentative else 0.85,
                      stroke_color=color, stroke_width=2.5 if tentative else 0).move_to([(x0 + x1) / 2, (ya + yb) / 2, 0])  # fmt: skip
        if tentative:
            r = VGroup(
                r.set_stroke(width=0),
                DashedVMobject(
                    r.copy().set_fill(opacity=0).set_stroke(color, 2.5), num_dashes=24
                ),
            )
        v = lab(signed(b - a), 22, INK, "SEMIBOLD").next_to(r, DOWN, buff=0.1)
        if sub:
            v = (
                VGroup(v, lab(sub, 18, MUTED))
                .arrange(DOWN, buff=0.06)
                .next_to(r, DOWN, buff=0.1)
            )
        n = lab(names[x] + ("\n(tentative)" if tentative else ""), 20, INK2).next_to(
            P.p(x, 1.246), DOWN, buff=0.12
        )
        return VGroup(r, v, n)


class M3(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Why the law missed")
        Bd = W["band"]
        c = np.array(Bd["c"])
        P = Plot((4e17, 1.5e21), (1.22, 1.40), 11.0, 4.5, (-0.3, 0.2), xt=[(10.0**e, f"10<sup>{e}</sup>") for e in (18, 19, 20, 21)],
                 yt=[(y, f"{y:.2f}") for y in (1.25, 1.30, 1.35)], xl="training compute (FLOPs)", yl="loss at the best size (nats)")  # fmt: skip
        td = trough_dots(P)
        cis = VGroup(
            *[
                Line(P.p(s["c"], lo), P.p(s["c"], hi), color=BUD[b], stroke_width=4)
                for b, (s, (lo, hi)) in enumerate(zip(SW, W["trough_ci"]))
            ]
        )
        lf = P.line(c, Bd["law"], ORANGE, 4)
        pl = P.line(c, Bd["troughs"], BLUE, 3, dash=40)
        ll = lab("the law", 21, ORANGE, "SEMIBOLD").next_to(
            P.p(1e21, Bd["law"][-1]), UP, buff=0.1
        )
        pll = lab("troughs", 21, BLUE, "SEMIBOLD").next_to(
            P.p(3e19, float(np.interp(np.log(3e19), np.log(c), Bd["troughs"]))),
            DOWN + LEFT,
            buff=0.1,
        )
        self.play(FadeIn(k), FadeIn(P), FadeIn(td), FadeIn(cis), Create(lf), Create(pl), FadeIn(ll), FadeIn(pll),
                  *cap("Skip the steep sides: fit only the three troughs, with the floor left free."))  # fmt: skip
        cap.hold()
        lt = ValueTracker(np.log10(c[0]) + 0.3)

        def band():
            m = c <= 10 ** lt.get_value()
            pts = [P.p(x, y) for x, y in zip(c[m], np.array(Bd["lo"])[m])] + [
                P.p(x, y) for x, y in zip(c[m][::-1], np.array(Bd["hi"])[m][::-1])
            ]
            return Polygon(*pts, fill_color=BLUE, fill_opacity=0.16, stroke_width=0)

        bd = always_redraw(band)
        self.add(bd)
        self.bring_to_back(bd)
        self.play(lt.animate(rate_func=linear).set_value(np.log10(c[-1])), run_time=3.0,
                  *cap("Every curve in this band fits them within noise. It fans out fast."))  # fmt: skip
        bd.clear_updaters()
        cap.hold()
        st = star(P.p(M["C20"], M["Y20"] - W["pen20"]), 0.2)
        sl = lab("Allie 2.0", 21, INK, "SEMIBOLD").next_to(st, UP, buff=0.1)
        self.play(
            FadeIn(st, scale=0.5),
            FadeIn(sl),
            *cap("Allie 2.0 sits inside it: three troughs cannot pin the floor."),
        )
        cap.hold()
        self.play(
            *cap(
                "The sweep pins how to grow the model; it cannot pin how low the loss goes."
            )
        )
        cap.hold(0.6)
        outro(self)


# ------------------------------------------------------------------ 9. the Allie 2.1 forecast


class S9(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Forecasting Allie 2.1")
        F = D["forecast"]
        P = frontier(center=(-1.05, 0.1), w=9.9)
        c20, c21 = M["C20"], M["C21"]
        td = trough_dots(P)
        lf, nf = law_frontier(P, 4, c21), nofloor(P, 3, c21)
        st = star(P.p(c20, M["Y20"]), 0.2)
        stl = lab("Allie 2.0", 20, INK).next_to(st, UP, buff=0.06)
        self.play(
            FadeIn(k),
            FadeIn(P),
            FadeIn(td),
            FadeIn(lf),
            FadeIn(nf),
            FadeIn(st),
            FadeIn(stl),
            *cap(f"Allie 2.1 trains with {c21 / c20:.1f} times Allie 2.0's compute."),
        )
        v21 = P.vline(c21, RULE, 2.5)
        l21 = lab(f"Allie 2.1\n{sci(c21)} FLOPs", 22, INK, "SEMIBOLD").next_to(
            P.p(c21, 1.39), DOWN + RIGHT, buff=0.1
        )
        self.play(Create(v21), FadeIn(l21))
        cap.hold()
        f21 = F["f21"]
        x = P.p(c21, 1.3)[0]
        spec = {"law": (ORANGE, "sweep law", 1.282), "shift": (GREY, "law shifted to 2.0", 1.262), "frontier4": (INK, "troughs + 2.0", 1.244),
                "power20": ("#6E6A63", "power law via 2.0", 1.226), "nofloor": (BLUE, "troughs, no floor", 1.200)}  # fmt: skip
        dd, tl = {}, {}
        for k_, (c, s, ly) in spec.items():
            dd[k_] = diamond([x, P.p(c21, f21[k_])[1], 0], c)
            t = lab(
                f"{f21[k_]:.3f}  {s}",
                21,
                INK if k_ == "frontier4" else (c if k_ in ("law", "nofloor") else INK2),
                "SEMIBOLD",
            )
            t.move_to([x + 0.55, P.p(c21, ly)[1], 0], aligned_edge=LEFT)
            lead = Line(
                dd[k_].get_center() + 0.12 * RIGHT,
                t.get_left() + 0.08 * LEFT,
                color=MUTED,
                stroke_width=1.5,
            )
            tl[k_] = VGroup(lead, t)
        self.play(*[GrowFromCenter(dd[k_]) for k_ in ("law", "nofloor")], *[FadeIn(tl[k_]) for k_ in ("law", "nofloor")],
                  *cap(f"Sweep-only fits disagree by {f21['law'] - f21['nofloor']:.2f}, with nothing to choose between them."))  # fmt: skip
        cap.hold()

        f4 = F["frontier4"]
        cs = np.logspace(np.log10(4.5e17), np.log10(c21), 80)
        anchor = P.line(cs, f4["E"] + f4["k"] * (cs / 1e18) ** -f4["gamma"], INK, 4)
        self.play(lf.animate.set_stroke(opacity=0.3), nf.animate.set_stroke(opacity=0.3), Create(anchor), run_time=1.6,
                  *cap("So anchor the level on Allie 2.0: refit the troughs with 2.0 as a fourth point."))  # fmt: skip
        self.play(GrowFromCenter(dd["frontier4"]), FadeIn(tl["frontier4"]))
        cap.hold()
        O = F["old"]
        y0, y1 = P.p(c21, O["lo"])[1], P.p(c21, O["hi"])[1]
        band = Rectangle(
            width=0.36,
            height=y1 - y0 + 0.2,
            fill_color=INK,
            fill_opacity=0.12,
            stroke_width=0,
        ).move_to([x, (y0 + y1) / 2, 0])
        self.play(FadeIn(band), *[GrowFromCenter(dd[k_]) for k_ in ("shift", "power20")], *[FadeIn(tl[k_]) for k_ in ("shift", "power20")],
                  *cap(f"Readings anchored on 2.0 span {O['lo']:.3f} to {O['hi']:.3f}."))  # fmt: skip
        self.bring_to_front(dd["frontier4"])
        cap.hold()
        self.play(
            *cap(
                "But none of them knows what else changed from 2.0 to 2.1, or how noisy one run is."
            )
        )
        cap.hold(0.6)
        outro(self)


class S10(Scene):
    def construct(self):
        paper(self)
        cap = Cap(self)
        k = kicker("Forecasting Allie 2.1")
        F = D["forecast"]
        O, y20 = F["old"], M["Y20"]
        P = Plot((-0.7, 8.5), (1.226, 1.259), 11.8, 4.3, (0.25, 0.35), logx=False,
                 yt=[(y, f"{y:.2f}") for y in (1.23, 1.24, 1.25)], yl="loss (nats)")  # fmt: skip

        def under(x, s, color=INK2):
            return lab(s, 20, color).next_to(P.p(x, 1.226), DOWN, buff=0.14)

        lvl = Line(P.p(-0.32, y20), P.p(0.32, y20), color=INK, stroke_width=6)
        l0 = lab(f"{y20:.4f}", 22, INK, "SEMIBOLD").next_to(lvl, UP, buff=0.1)
        n0 = under(0, "Allie 2.0\nbig run", INK)
        self.play(FadeIn(k), FadeIn(P), Create(lvl), FadeIn(l0), FadeIn(n0),
                  *cap(f"Instead, start from Allie 2.0's big run, {y20:.3f}, and add each change."))  # fmt: skip
        cap.hold()

        def step(i, a, st):
            b = a + st["d"]
            u = 1.3 * i
            x0, x1 = P.p(u - 0.28, 0)[0], P.p(u + 0.28, 0)[0]
            ya, yb = P.p(0, a)[1], P.p(0, b)[1]
            col = BLUE if st["d"] < 0 else ORANGE
            g = VGroup(DashedLine(P.p(u - 1.3 + 0.32, a), P.p(u - 0.28, a), color=MUTED, stroke_width=2, dash_length=0.06),
                       Rectangle(width=x1 - x0, height=max(abs(ya - yb), 0.03), fill_color=col, fill_opacity=0.8,
                                 stroke_width=0).move_to([(x0 + x1) / 2, (ya + yb) / 2, 0]))  # fmt: skip
            if st["ci"]:
                cx = (x0 + x1) / 2
                c0, c1 = P.p(0, a + st["ci"][0])[1], P.p(0, a + st["ci"][1])[1]
                g.add(VGroup(Line([cx, c0, 0], [cx, c1, 0]), Line([cx - 0.07, c0, 0], [cx + 0.07, c0, 0]),
                             Line([cx - 0.07, c1, 0], [cx + 0.07, c1, 0])).set_stroke(INK, 2.5))  # fmt: skip
            v = lab(signed(st["d"], 4), 22, INK, "SEMIBOLD").next_to(
                g[1:], UP, buff=0.1
            )
            return VGroup(g, v, under(u, st["name"])), b

        a, steps = y20, []
        for i, st in enumerate(F["steps"], 1):
            g, a = step(i, a, st)
            steps.append(g)
        s0 = F["steps"][0]
        self.play(GrowFromEdge(steps[0][0][1], UP), FadeIn(steps[0][0][0]), FadeIn(steps[0][0][2:]), FadeIn(steps[0][1:]),
                  *cap(f"The sweep gives the step to a bigger model on more tokens: {signed(s0['d'])}."))  # fmt: skip
        cap.hold()
        self.play(LaggedStart(*[FadeIn(g, shift=0.1 * DOWN) for g in steps[1:]], lag_ratio=0.4), run_time=1.6,
                  *cap("Then 2.1's longer decay, its new router, and its 112B-token table's small cost."))  # fmt: skip
        cap.hold()

        xf = 6.6
        cx = P.p(xf, 0)[0]
        f0, f1 = P.p(0, F["lo"])[1], P.p(0, F["hi"])[1]
        ci = VGroup(Line([cx, f0, 0], [cx, f1, 0], stroke_width=7), Line([cx - 0.12, f0, 0], [cx + 0.12, f0, 0], stroke_width=4),
                    Line([cx - 0.12, f1, 0], [cx + 0.12, f1, 0], stroke_width=4)).set_color(INK)  # fmt: skip
        fd = diamond(P.p(xf, a), INK, 0.14)
        conn = DashedLine(
            P.p(1.3 * len(F["steps"]) + 0.28, a),
            P.p(xf - 0.2, a),
            color=MUTED,
            stroke_width=2,
            dash_length=0.06,
        )
        fl = lab(f"{F['central']:.3f}", 30, INK, "SEMIBOLD").next_to(
            fd, RIGHT, buff=0.2
        )
        fn = under(xf, "Allie 2.1\nbig run", INK)
        cl = lab(f"90%: {F['lo']:.3f}–{F['hi']:.3f}", 20, INK2).next_to(
            ci, UP, buff=0.1
        )
        self.play(Create(conn), GrowFromCenter(fd), FadeIn(fl), FadeIn(fn),
                  *cap(f"Allie 2.1's big run: about {F['central']:.3f}, {y20 - F['central']:.3f} below 2.0."))  # fmt: skip
        cap.hold()
        self.add(ci)
        self.bring_to_front(fd)
        self.play(GrowFromCenter(ci), FadeIn(cl),
                  *cap(f"Add the steps' spread and ±{F['noise']:.3f} of run noise: 90% between {F['lo']:.3f} and {F['hi']:.3f}."))  # fmt: skip
        cap.hold()
        xo = 7.8
        ox = P.p(xo, 0)[0]
        o0, o1 = P.p(0, O["lo"])[1], P.p(0, O["hi"])[1]
        old = VGroup(Line([ox, o0, 0], [ox, o1, 0], stroke_width=7), Line([ox - 0.1, o0, 0], [ox + 0.1, o0, 0], stroke_width=4),
                     Line([ox - 0.1, o1, 0], [ox + 0.1, o1, 0], stroke_width=4)).set_color(MUTED)  # fmt: skip
        on = under(xo, "earlier\nreadings", MUTED)
        self.play(
            GrowFromCenter(old),
            FadeIn(on),
            *cap(
                f"Wider than the earlier readings' {O['lo']:.3f}–{O['hi']:.3f}: they left out these steps and noise."
            ),
        )
        cap.hold()
        self.play(
            *cap(
                f"The compute-optimal split at 2.1's compute would be {size(LAW['nopt21'])} active on {LAW['dopt21'] / 1e9:.0f}B tokens."
            )
        )
        cap.hold()

        self.play(
            *[FadeOut(m) for m in self.mobjects if m is not cap.cur and m is not k]
        )
        N = F["next"]
        head = text("How we forecast now", 44, INK, weight="SEMIBOLD")
        rules = ("Anchor the level on the biggest real run; the sweep gives the step from it.",
                 "Report the spread across law forms and the troughs' band, not one fit's error bar.",
                 f"Train the sweep the way the big run trains: decay to {D['miss']['lr_end_20']:.1%} of peak.",
                 f"Before the next big run: {N['count']} long runs, {size(N['n'][0])}–{size(N['n'][1])} active, "
                 f"{N['tpp'][0]}–{N['tpp'][1]} tokens per parameter.")  # fmt: skip
        items = VGroup(
            *[
                VGroup(Dot(radius=0.06, color=ORANGE), text(r, 28, INK2)).arrange(
                    RIGHT, buff=0.25
                )
                for r in rules
            ]
        )
        items.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        VGroup(head, items).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(
            [0, 0.3, 0]
        )
        self.play(
            Write(head),
            *cap(
                f"First real check: 2.1's loss at the same point in its run as 2.0's, from about {N['check_from']:.0%} on."
            ),
        )
        cap.hold()
        self.play(
            *cap(
                f"Earlier checkpoints say little: they carry at least ±{N['ckpt_noise']:.3f} of noise."
            )
        )
        self.play(
            LaggedStart(*[FadeIn(r, shift=0.1 * UP) for r in items], lag_ratio=0.5),
            run_time=2.4,
        )
        cap.hold(1.5)
        outro(self, 1.2)
