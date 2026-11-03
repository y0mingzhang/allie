"""Allie's scaling law: what one is for, how its forecast missed Allie 2.0's big run, what that changed for Allie 2.1.

Every number drawn is in data.json, with its source (build_data.py). render.sh renders VIDEO's scenes and joins them.
"""

import json
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    Create,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    ReplacementTransform,
    Scene,
    Square,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402
from style import (  # noqa: E402
    ALLIE20,
    ALLIE21,
    BODY,
    DENSE,
    DISPLAY,
    GROUND,
    HEAD,
    HILITE,
    INK,
    INK2,
    LABEL,
    MARGIN,
    MONO,
    MUTED,
    PANEL,
    ROSE,
    RULE,
    SEQ,
    SKY,
    SLAB,
    SMALL,
    T_BEAT,
    T_MOVE,
    T_QUICK,
    T_SLOW,
    TITLE,
)

D = json.loads(Path(__file__).with_name("data.json").read_text())
F, FR, FC, RUN, BT, FO, FAN, CK, RF, A21, HB, DEMO = (
    D[k] for k in "first frontier forecast20 run20 backtest forms20 fan checks refit allie21 how_big demo".split()
)
LAW, B = F["law"], F["budgets"]
C20, C21, TOP = FC["c"], RF["c21"], B[-1]["c"]
BUD, BUD4 = SEQ[2:5], SEQ[1:5]
FOUR = ("additive", "coupled", "n_dep_data", "two_d_terms")
LO21, HI21 = min(RF["forms"][k]["at21"] for k in FOUR), max(RF["forms"][k]["at21"] for k in FOUR)
ANCHOR = max(a["c"] for a in RF["anchors"])
XR = (4e17, 1.5e21)
N_MOE, N_DENSE = sum(r["moe"] for r in F["runs"]), sum(not r["moe"] for r in F["runs"])
HALF = (FC["ci90"][1] - FC["ci90"][0]) / 2
LR = RF["lr21"]
GAPS = LR["lo"] - A21["sept"], LR["hi"] - A21["sept"]

VIDEO = dict(
    scenes=["Open", "Budget", "Formula", "Forecast", "TheRun", "Checks", "Why", "Second", "HowBig", "Lessons"],
    gif=("TheRun", 0.5, 22.0),
    poster=("Lessons", 4.0),
)


def L(s, px=LABEL, color=INK, **kw):
    return S.text(s, px, color, **kw)


def T(s, px=LABEL, color=INK, **kw):
    return S.text(s, px, color, markup=True, **kw)


def stack(*ms, buff=0.08, edge=LEFT):
    return VGroup(*ms).arrange(DOWN, buff=buff, aligned_edge=edge)


def fade(*ms):
    return FadeIn(VGroup(*ms))


def sci(v):
    e = int(np.floor(np.log10(v)))
    return f"{v / 10**e:.1f}×10<sup>{e}</sup>"


def kappa(n):
    return np.interp(np.log(n), F["kappa"]["lnn"], F["kappa"]["k"])


def size_term(n):
    return LAW["A"] * (n / 1e7) ** -LAW["alpha"]


def data_term(d):
    return LAW["B"] * (d / 1e8) ** -LAW["beta"]


def iso(n, c):
    return LAW["E"] + size_term(n) + data_term(c / (kappa(n) * n))


def clear(scene, t=T_MOVE):
    for m in scene.mobjects:
        m.clear_updaters()
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=t)


def lr_tag():
    return S.tag(
        f"assumes the law's best rate, {LR['best']:.2f}×; 2.1 ran {LR['ran']:.2f}×, which that law prices at +{LR['price']:.3f}"
    )


def bracket(x, y0, y1, color=HILITE):
    ends = ([x, y0, 0], [x, y1, 0], [x - 0.1, y0, 0], [x - 0.1, y1, 0])
    return (
        VGroup(Line(ends[0], ends[1]), Line(ends[0], ends[2]), Line(ends[1], ends[3]))
        .set_color(color)
        .set_stroke(width=3)
    )


def bar(w, h, color, **kw):
    return Rectangle(width=w, height=h, fill_color=color, fill_opacity=1, stroke_width=0, **kw)


class Plot(VGroup):
    """Log or linear x, linear y, horizontal gridlines; p() maps data to the frame."""

    def __init__(self, xr, yr, box, xt=(), yt=(), xl=None, yl=None, logx=True):
        super().__init__()
        self.xr, self.yr, self.box, self.logx = xr, yr, box, logx
        x0, y0, x1, y1 = box
        self.add(*[Line(self.p(xr[0], v), self.p(xr[1], v), color=RULE, stroke_width=1) for v, _ in yt])
        self.add(Line([x0, y0, 0], [x1, y0, 0], color=RULE, stroke_width=2.5))
        self.add(*[L(s, SMALL, MUTED).next_to(self.p(xr[0], v), LEFT, buff=0.14) for v, s in yt])
        self.add(*[T(s, SMALL, MUTED).move_to([self.fx(v), y0 - 0.28, 0]) for v, s in xt])
        if xl:
            self.add(L(xl, SMALL, INK2).move_to([(x0 + x1) / 2, y0 - 0.62, 0]))
        if yl:
            t = L(yl, SMALL, INK2)
            self.add(t.move_to([x0 - 0.62 + t.width / 2, y1 + 0.28, 0]))

    def fx(self, x):
        a, b = self.xr
        u = (np.log(x) - np.log(a)) / (np.log(b) - np.log(a)) if self.logx else (x - a) / (b - a)
        return self.box[0] + u * (self.box[2] - self.box[0])

    def fy(self, y):
        a, b = self.yr
        return self.box[1] + (y - a) / (b - a) * (self.box[3] - self.box[1])

    def p(self, x, y):
        return np.array([self.fx(x), self.fy(y), 0.0])

    def curve(self, xs, ys, color, w=4.0, dash=False, o=1.0):
        m = VMobject(stroke_color=color, stroke_width=w, stroke_opacity=o, fill_opacity=0)
        m.set_points_as_corners([self.p(x, y) for x, y in zip(xs, ys)])
        return DashedVMobject(m, num_dashes=max(6, int(m.get_arc_length() / 0.17)), dashed_ratio=0.55) if dash else m

    def dot(self, x, y, color, r=0.075):
        return Dot(self.p(x, y), radius=r, color=color)

    def hollow(self, x, y, color, r=0.12, w=4.5):
        return Circle(radius=r, color=color, stroke_width=w).move_to(self.p(x, y)).set_fill(GROUND, 1)

    def ring(self, x, y, color=INK, r=0.17, w=3.0):
        return Circle(radius=r, color=color, stroke_width=w).move_to(self.p(x, y))

    def hline(self, y, color=INK2, w=2.5):
        return self.curve(self.xr, [y, y], color, w, dash=True)

    def vline(self, x, color=INK2, w=2.0):
        return self.curve([x, x], self.yr, color, w, dash=True)

    def gap(self, x0, x1):
        a, b, y0, y1 = self.fx(x0), self.fx(x1), self.box[1], self.box[3]
        return bar(b - a, y1 - y0, PANEL).move_to([(a + b) / 2, (y0 + y1) / 2, 0])

    def whisker(self, x, lo, hi, color, w=4.5, cap=0.09):
        a, b = self.p(x, lo), self.p(x, hi)
        parts = (Line(a, b), Line(a + cap * LEFT, a + cap * RIGHT), Line(b + cap * LEFT, b + cap * RIGHT))
        return VGroup(*parts).set_color(color).set_stroke(width=w)


# ------------------------------------------------------------------ the question


class Open(Scene):
    def construct(self):
        S.ground(self)
        self.cap = S.Captions(self)
        t = L("How good will it be?", DISPLAY, font=SLAB, weight="SEMIBOLD").to_edge(LEFT, buff=MARGIN).shift(0.6 * UP)
        sub = L("Allie's scaling law: a forecast, a miss, and what the miss taught us", BODY, INK2)
        self.play(fade(t, sub.next_to(t, DOWN, buff=0.4, aligned_edge=LEFT)), run_time=T_MOVE)
        self.wait(1.8)
        self.play(FadeOut(t), FadeOut(sub), run_time=T_QUICK)
        self.add(S.kicker("The question"))
        self.loss()
        self.cost()

    def loss(self):
        cap = self.cap
        b = S.board(DEMO["fen"], size=4.3).move_to([-4.2, 0.55, 0])
        x_san, x_bar, width, x_s, y0, dy = -0.95, -0.25, 3.5, 6.0, 1.9, 0.72
        rows = VGroup()
        for i, m in enumerate(DEMO["moves"]):
            y = y0 - i * dy
            w = bar(max(width * m["p"], 0.03), 0.42, ALLIE21).move_to([x_bar, y, 0], aligned_edge=LEFT)
            pct = f"{m['p']:.0%}" if m["p"] >= 0.1 else f"{m['p']:.1%}"
            rows.add(
                VGroup(
                    L(m["san"], font=MONO).move_to([x_san, y, 0]),
                    w,
                    L(pct, LABEL, INK2).next_to(w, RIGHT, buff=0.12),
                    L(f"{m['surprise']:.2f}", font=MONO).move_to([x_s, y, 0]),
                )
            )
        h1 = L("Allie 2.1's chance for each move, 1800 v 1750, 3+2", SMALL, INK2).move_to(
            [x_bar, y0 + 0.62, 0], aligned_edge=LEFT
        )
        h2 = L("surprise, −ln p", SMALL, INK2).move_to([6.6, y0 + 0.62, 0], aligned_edge=RIGHT)
        p0, p2 = DEMO["moves"][0], DEMO["moves"][2]
        cap.say(
            f"Allie predicts the move a person will play. Here: Nc6, {p0['p']:.0%}.",
            fade(b, h1, *[r[:3] for r in rows]),
        )
        mark = lambda i: Rectangle(width=7.5, height=0.6, stroke_color=HILITE, stroke_width=3).move_to(
            [2.5, y0 - i * dy, 0]
        )
        m0, m2 = mark(0), mark(2)
        say = f"Its loss is its surprise at the move really played: {p0['surprise']:.2f} if Nc6, {p2['surprise']:.2f} if d6. Lower is better."
        self.play(fade(h2, *[r[3] for r in rows], m0), *cap(say), run_time=T_MOVE)
        self.wait(1.8)
        self.play(ReplacementTransform(m0, m2), run_time=T_MOVE)
        self.wait(1.2)
        avg = VGroup(
            L(f"mean of 16 time-control and rating cells, {CK['moves'] / 1e6:.2f}M moves:", LABEL, INK2),
            L(f"{A21['main']:.4f}", HEAD, weight="SEMIBOLD"),
        ).arrange(RIGHT, buff=0.3)
        self.play(FadeOut(m2), FadeIn(avg.move_to([2.5, -2.0, 0])), run_time=T_MOVE)
        cap.hold(0.5)
        self.play(FadeOut(VGroup(b, h1, h2, rows, avg)), run_time=T_QUICK)

    def cost(self):
        cap = self.cap
        side = 3.9
        big = Square(side, fill_color=ALLIE20, fill_opacity=1, stroke_width=0).move_to([4.4, 0.45, 0])
        bl, bv = (
            L("Allie 2.0's big run").next_to(big, UP, buff=0.14),
            T(f"{sci(C20)} FLOPs", SMALL, INK2).next_to(big, DOWN, buff=0.14),
        )
        cap.say(f"Allie 2.0's big run took {RUN['days']} days on {RUN['gpus']} GPUs.", fade(big, bl, bv))
        rows, small = VGroup(), VGroup()
        for b, y in zip(("1e18", "3e17", "1e17"), (1.75, 0.45, -0.72)):
            rs = [r for r in F["runs"] if r["budget"] == b]
            sq = VGroup(
                *[Square(side * np.sqrt(r["c"] / C20), fill_color=BUD[1], fill_opacity=1, stroke_width=0) for r in rs]
            )
            small.add(sq.arrange(RIGHT, buff=0.06).move_to([-6.6, y, 0], aligned_edge=LEFT))
            rows.add(
                T(f"{len(rs)} runs at {sci(rs[0]['c'])} FLOPs each", SMALL, INK2).next_to(
                    sq, UP, buff=0.1, aligned_edge=LEFT
                )
            )
        area = S.tag("area is compute, counted in arithmetic operations: FLOPs").move_to(
            [-6.6, -1.95, 0], aligned_edge=LEFT
        )
        say = f"A sweep of {len(F['runs'])} smaller runs came first. Together they cost a third of it."
        self.play(
            LaggedStart(*[FadeIn(s) for g in small for s in g], lag_ratio=0.03),
            fade(rows, area),
            *cap(say),
            run_time=T_SLOW,
        )
        third = bar(side * F["compute_all"] / C20, side, BUD[1]).align_to(big, LEFT).align_to(big, DOWN)
        self.play(ReplacementTransform(small.copy(), third), run_time=T_SLOW)
        self.play(FadeIn(tl := L(f"all {len(F['runs'])}", LABEL, GROUND).move_to(third)), run_time=T_QUICK)
        cap.hold()
        self.play(FadeOut(VGroup(big, bl, bv, rows, small, area, third, tl)), run_time=T_QUICK)
        q = [
            L(s, TITLE, font=SLAB, weight="SEMIBOLD") for s in ("How big should the model be?", "How good will it get?")
        ]
        VGroup(*q).arrange(DOWN, buff=0.45, aligned_edge=LEFT).to_edge(LEFT, buff=MARGIN).shift(0.5 * UP)
        self.play(
            FadeIn(q[0]),
            *cap(
                "You can't try many sizes at that cost. A scaling law tries to answer both questions from small runs."
            ),
            run_time=T_MOVE,
        )
        self.play(FadeIn(q[1]), run_time=T_MOVE)
        cap.hold(T_BEAT)
        clear(self)


# ------------------------------------------------------------------ what a budget buys


class Budget(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        b = B[1]
        pts = b["points"]
        ox, oy = -5.85, -1.85
        sx, sy = 4.6 / pts[-1]["n"], 4.5 / pts[0]["d"]

        def rect(p):
            return bar(p["n"] * sx, p["d"] * sy, BUD[1]).move_to([ox, oy, 0], aligned_edge=DOWN + LEFT)

        def vals(p):
            r = rect(p)
            return VGroup(
                L(f"{p['n'] / 1e6:.0f}M").next_to(r.get_corner(DOWN + RIGHT), DOWN, buff=0.1),
                L(f"{p['d'] / 1e9:.1f}B").next_to(r.get_corner(UP + LEFT), LEFT, buff=0.1),
            )

        ax = VGroup(
            Line([ox, oy, 0], [ox + 4.8, oy, 0], color=RULE, stroke_width=2.5),
            Line([ox, oy, 0], [ox, oy + 4.7, 0], color=RULE, stroke_width=2.5),
        )
        xn = L("parameters used per move", SMALL, INK2).move_to([ox + 2.4, oy - 0.62, 0])
        yn = L("training tokens, about one per move", SMALL, INK2).move_to([-6.6, oy + 4.98, 0], aligned_edge=LEFT)
        note = stack(L("area is compute:", SMALL, INK2), T(f"{sci(b['c'])} FLOPs", SMALL)).move_to(
            [ox + 2.4, oy + 4.05, 0], aligned_edge=LEFT
        )
        P = Plot((2.2e7, 4.6e8), (1.33, 1.455), (0.05, -1.85, 6.4, 2.75), xt=[(3e7, "30M"), (1e8, "100M"), (3e8, "300M")],
                 yt=[(1.35, "1.35"), (1.40, "1.40"), (1.45, "1.45")], xl="parameters used per move", yl="loss on the main test set")  # fmt: skip
        r, v = rect(pts[2]), vals(pts[2])
        self.add(S.kicker("What a budget buys"))
        cap.say(
            "A compute budget buys parameters times training tokens. A bigger model sees fewer moves.",
            fade(ax, xn, yn, note, r, v),
        )
        d = P.dot(pts[0]["n"], pts[0]["loss"], BUD[1], 0.09)
        r2, v2 = rect(pts[0]), vals(pts[0])
        self.play(
            ReplacementTransform(r, r2),
            FadeOut(v),
            FadeIn(v2),
            FadeIn(P),
            FadeIn(d),
            *cap("Train a model at each split and score it."),
            run_time=T_MOVE,
        )
        r, v = r2, v2
        for p in pts[1:]:
            r2, v2 = rect(p), vals(p)
            self.play(
                ReplacementTransform(r, r2),
                FadeOut(v),
                FadeIn(v2),
                FadeIn(P.dot(p["n"], p["loss"], BUD[1], 0.09), scale=1.6),
                run_time=0.55,
            )
            r, v = r2, v2
        cap.hold()
        small = L("too small", LABEL, INK2).next_to(P.p(pts[0]["n"], pts[0]["loss"]), UP, buff=0.32).shift(0.05 * RIGHT)
        big = L("too few tokens", LABEL, INK2).move_to(P.p(1.7e8, 1.425))
        self.play(
            FadeIn(small), *cap("Too small, and it runs out of capacity. Too big, out of tokens."), run_time=T_MOVE
        )
        self.wait(1.6)
        self.play(FadeIn(big), run_time=T_MOVE)
        cap.hold()
        xs = np.geomspace(pts[1]["n"] / 1.12, pts[4]["n"] * 1.12, 50)
        ring = P.ring(b["n_opt"], b["l_opt"])
        rl = L(f"best: {b['n_opt'] / 1e6:.0f}M").next_to(ring, DOWN, buff=0.12)
        say = "Loss against size is a U. Its bottom is the best this budget can buy."
        cap.say(
            say, Create(P.curve(xs, np.polyval(b["parabola"], np.log(xs)), BUD[1], 4)), fade(ring, rl), run_time=T_SLOW
        )
        clear(self)


# ------------------------------------------------------------------ one formula


class Formula(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        P = Plot((1.2e7, 5e8), (1.25, 1.54), (-5.7, -1.85, 6.4, 1.8), xt=[(3e7, "30M"), (1e8, "100M"), (3e8, "300M")],
                 yt=[(1.30, "1.30"), (1.40, "1.40"), (1.50, "1.50")], xl="parameters used per move", yl="loss")  # fmt: skip
        self.add(
            S.kicker("One formula"),
            L("in the style of Chinchilla (2022)", SMALL, MUTED).move_to([6.6, S.KICKER_Y, 0], aligned_edge=RIGHT),
        )
        dots, us, rings = VGroup(), VGroup(), VGroup()
        for i, b in enumerate(B):
            dots.add(VGroup(*[P.dot(p["n"], p["loss"], BUD[i], 0.085) for p in b["points"]]))
            win = [p["n"] for p in b["points"] if p["window"]]
            xs = np.geomspace(win[0] / 1.12, win[-1] * 1.12, 50)
            us.add(P.curve(xs, np.polyval(b["parabola"], np.log(xs)), BUD[i], 3.5))
            rings.add(P.ring(b["n_opt"], b["l_opt"]))
        self.play(fade(P, dots[1], us[1], rings[1]), run_time=T_MOVE)
        labels = VGroup(*[T(f"{sci(b['c'])} FLOPs", SMALL, INK2) for b in B])
        for m, xy in zip(labels, ((1.8e7, 1.413), (2.45e8, 1.465), (3.1e8, 1.313))):
            m.move_to(P.p(*xy))
        dense = VGroup(
            *[P.dot(r["n"], r["loss"], DENSE, 0.055) for r in F["runs"] if not r["moe"] and r["loss"] < 1.54]
        )
        tag = S.tag(
            f"the other {N_DENSE} runs: plain dense models, which lost at each budget's best size, in counted FLOPs"
        ).move_to(P.p(1.25e7, 1.51), aligned_edge=LEFT)
        cap.say(
            "Spend more, and the U moves down and to the right.",
            fade(dots[0], us[0], rings[0], dots[2], us[2], rings[2], labels, dense, tag),
            run_time=T_SLOW,
            extra=1.6,
        )
        self.play(FadeOut(VGroup(dense, tag, us, rings)), run_time=T_QUICK)

        xpos = (-3.9, -2.75, -1.45, -0.15, 1.65, 3.45, 5.25)
        heads = ("loss", "=", "floor", "+", "size penalty", "+", "data penalty")
        words = VGroup(
            *[
                L(s, BODY, INK2 if s in "=+" else INK, weight="MEDIUM" if s in "=+" else "SEMIBOLD").move_to(
                    [x, 2.98, 0]
                )
                for s, x in zip(heads, xpos)
            ]
        )
        for m in words[::2]:
            m.align_to([0, 3.12, 0], UP)
        forms = (
            "L(N, D)",
            f"{LAW['E']:.3f}",
            f"{LAW['A']:.3f} (N/10<sup>7</sup>)<sup>−{LAW['alpha']:.2f}</sup>",
            f"{LAW['B']:.2f} (D/10<sup>8</sup>)<sup>−{LAW['beta']:.2f}</sup>",
        )
        syms = VGroup(*[T(s, LABEL, INK2).move_to([xpos[i], 2.52, 0]) for i, s in zip((0, 2, 4, 6), forms)])
        curves = VGroup()
        for b in B:
            xs = np.geomspace(1.25e7, 4.9e8, 80)
            ys = iso(xs, b["c"])
            curves.add(P.curve(xs[ys < 1.54], ys[ys < 1.54], SKY, 4))
        cap.say(
            f"A scaling law is one formula for all of it, fit to the {N_MOE} runs of Allie's design.",
            fade(words, syms),
            Create(curves),
            run_time=T_SLOW,
        )

        t = ValueTracker(np.log(1.5e7))
        c0 = B[0]["c"]

        def terms():
            n = np.exp(t.get_value())
            return n, size_term(n), data_term(c0 / (kappa(n) * n))

        def nums():
            _, a, d = terms()
            return VGroup(
                *[
                    L(f"{v:.3f}", font=MONO).move_to([xpos[i], 2.08, 0])
                    for i, v in zip((0, 2, 4, 6), (LAW["E"] + a + d, LAW["E"], a, d))
                ]
            )

        live, mark = (
            always_redraw(nums),
            always_redraw(lambda: Dot(P.p(terms()[0], iso(terms()[0], c0)), radius=0.11, color=HILITE)),
        )
        self.play(
            fade(live, mark),
            *cap("A bigger model cuts the size penalty, but sees fewer tokens. That trade is the U."),
            run_time=T_MOVE,
        )
        self.play(t.animate(rate_func=linear).set_value(np.log(1.77e8)), run_time=3.6)
        cap.hold()
        fl = L(f"floor {LAW['E']:.3f}").next_to(P.p(1.25e7, LAW["E"]), UP, buff=0.08, aligned_edge=LEFT)
        ci = S.tag(f"floor's 90% interval: {FC['floor_ci'][0]:.2f} to {FC['floor_ci'][1]:.2f}").move_to(
            P.p(1.25e7, 1.515), aligned_edge=LEFT
        )
        fit = S.tag(f"typical error {LAW['rmse']:.4f}, seed noise {F['sigma']:.4f}").next_to(
            ci, DOWN, buff=0.1, aligned_edge=LEFT
        )
        cap.say(
            "Nothing cuts the floor: the loss it expects with unlimited compute. The law fits within the noise between repeat runs.",
            FadeOut(mark),
            Create(P.hline(LAW["E"])),
            fade(fl, ci, fit),
            extra=1.0,
        )
        clear(self)


# ------------------------------------------------------------------ the forecast and the run


def canvas(
    box=(-5.7, -1.85, 6.4, 2.95),
    yr=(1.22, 1.41),
    yl="loss on the main test set, July 2026 games",
    xl="training compute (FLOPs)",
):
    yt = [(v, f"{v:.2f}") for v in (1.25, 1.30, 1.35, 1.40) if yr[0] < v < yr[1]]
    return Plot(XR, yr, box, xt=[(10.0**e, f"10<sup>{e}</sup>") for e in (18, 19, 20, 21)], yt=yt, xl=xl, yl=yl)


def law_line(P):
    c, y = np.array(FR["c"]), np.array(FR["loss"])
    at = lambda v: np.interp(np.log(v), np.log(c), y)
    near, far = c <= TOP, (c > TOP) & (c < C20)
    return P.curve(np.r_[c[near], TOP], np.r_[y[near], at(TOP)], SKY, 4.5), P.curve(
        np.r_[TOP, c[far], C20], np.r_[at(TOP), y[far], at(C20)], SKY, 4, dash=True
    )


def run_end(P):
    end = P.dot(C20, RUN["final"], ALLIE20, 0.14)
    return end, L(f"big run {RUN['final']:.4f}").next_to(end, RIGHT, buff=0.34)


def first_state(P):
    """The first forecast on its canvas, as Forecast leaves it."""
    solid, dash = law_line(P)
    dots = VGroup(*[P.dot(b["c"], b["l_opt"], BUD[i], 0.11) for i, b in enumerate(B)])
    fd = P.hollow(C20, FC["f"], SKY)
    who = stack(
        L("Allie 2.0's big run"),
        L(f"{FC['n'] / 1e9:.2f}B parameters", SMALL, INK2),
        L(f"{FC['d'] / 1e9:.0f}B tokens", SMALL, INK2),
        buff=0.05,
    )
    return dict(
        gap=P.gap(TOP, C20),
        P=P,
        solid=solid,
        dash=dash,
        dots=dots,
        tags=VGroup(
            *[L(f"{b['n_opt'] / 1e6:.0f}M", SMALL, INK2).next_to(dots[i], UP, buff=0.12) for i, b in enumerate(B)]
        ),
        gl=L("no runs here", LABEL, INK2).move_to(P.p(np.sqrt(TOP * C20), 1.335)),
        floor=P.hline(LAW["E"]),
        fl=L(f"floor {LAW['E']:.3f}").next_to(P.p(XR[0], LAW["E"]), UP, buff=0.08, aligned_edge=LEFT),
        v=P.vline(C20),
        vl=who.next_to(P.p(C20, 1.41), RIGHT, buff=0.14).shift(0.38 * DOWN),
        w=P.whisker(C20, *FC["ci90"], SKY),
        fd=fd,
        ft=stack(
            L(f"forecast {FC['f']:.4f}"), L(f"90%: {FC['ci90'][0]:.3f} to {FC['ci90'][1]:.3f}", SMALL, INK2), buff=0.05
        )
        .next_to(fd, RIGHT, buff=0.18)
        .shift(0.28 * UP),
    )


class Forecast(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        st = first_state(canvas())
        self.add(S.kicker("The forecast"))
        cap.say(
            "Against compute, the law is a curve through each budget's best size.",
            fade(st["P"], st["dots"], st["tags"], st["fl"]),
            Create(st["solid"]),
            Create(st["floor"]),
            run_time=T_SLOW,
        )
        self.add(st["gap"])
        self.bring_to_back(st["gap"])
        cap.say(
            f"The big run, then the largest model that fit on one {RUN['gpus']}-GPU machine, went {FC['distance']:.0f} times past the sweep.",
            FadeIn(st["gap"]),
            Create(st["v"]),
            Create(st["dash"]),
            fade(st["vl"], st["gl"]),
            run_time=T_SLOW,
        )
        cap.say(
            f"The law's forecast: {FC['f']:.4f}, give or take {HALF:.3f}.",
            fade(st["w"], st["fd"], st["ft"]),
            extra=T_BEAT,
        )


class TheRun(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        P = canvas()
        st = first_state(P)
        self.add(S.kicker("The run"), *st.values())
        tok, c, y = (np.array(RUN[k]) for k in ("tokens", "c", "loss"))
        t = ValueTracker(tok[0])

        def line():
            u = t.get_value()
            m = tok <= u
            return P.curve(np.r_[c[m], np.interp(u, tok, c)], np.r_[y[m], np.interp(u, tok, y)], ALLIE20, 5)

        def ticker():
            u = t.get_value()
            return stack(
                L(f"{u / 1e9:.1f}B of {FC['d'] / 1e9:.0f}B tokens", LABEL, INK2),
                L(f"loss {np.interp(u, tok, y):.4f}", HEAD, weight="SEMIBOLD"),
                buff=0.1,
            ).move_to([-3.1, 2.2, 0], aligned_edge=LEFT)

        ln, tk = always_redraw(line), always_redraw(ticker)
        hd = always_redraw(
            lambda: P.dot(np.interp(t.get_value(), tok, c), np.interp(t.get_value(), tok, y), ALLIE20, 0.11)
        )
        self.play(
            fade(ln, hd, tk), *cap("Then we ran it. The line is its checkpoints, on the same test."), run_time=T_MOVE
        )
        pf, pl = RUN["pass_forecast"], RUN["pass_floor"]
        self.play(t.animate(rate_func=linear).set_value(pf["tokens"]), run_time=4.6)
        r1 = P.ring(np.interp(pf["tokens"], tok, c), FC["f"], HILITE, 0.2, 4)
        cap.say(f"About {pf['frac']:.0%} of the way through, it passed the forecast.", FadeIn(r1, scale=1.6))
        self.play(FadeOut(r1), t.animate(rate_func=linear).set_value(pl["tokens"]), run_time=1.8)
        r2 = P.ring(np.interp(pl["tokens"], tok, c), LAW["E"], HILITE, 0.2, 4)
        cap.say(
            f"At about {pl['frac']:.0%}, it passed the floor: the level no compute was supposed to reach.",
            FadeIn(r2, scale=1.6),
        )
        self.play(FadeOut(r2), t.animate(rate_func=linear).set_value(tok[-1]), run_time=1.6)
        for m in (ln, hd, tk):
            m.clear_updaters()
        end, el = run_end(P)
        rel = (
            S.tag(f"released after a short second anneal: {RUN['released']:.4f}")
            .move_to(P.p(C20, 1.2315), aligned_edge=RIGHT)
            .shift(0.12 * LEFT)
        )
        bx = P.fx(C20) + 0.42
        br = bracket(bx, P.fy(FC["f"] - 0.0035), P.fy(RUN["final"] + 0.0035))
        lab = VGroup(
            L(f"{RUN['miss']:.3f}", BODY, HILITE, weight="SEMIBOLD").next_to(br, RIGHT, buff=0.08).set_y(P.fy(1.279)),
            L(f"{RUN['noise_multiple']:.0f}× seed noise", SMALL, INK2)
            .next_to(br, RIGHT, buff=0.08)
            .set_y(P.fy(1.2655)),
        )
        gains = VGroup()
        for s, v, col in (
            ("the law promised", FC["gain_promised"], SKY),
            ("the run delivered", FC["gain_real"], ALLIE20),
        ):
            row = VGroup(bar(4.2 * v / FC["gain_real"], 0.34, col), L(f"{v:.3f}")).arrange(RIGHT, buff=0.12)
            gains.add(stack(L(s, LABEL, INK2), row))
        gains.arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([-3.1, 2.1, 0], aligned_edge=LEFT)
        say = f"It finished at {RUN['final']:.4f}, {RUN['miss']:.3f} below the forecast. Past the sweep's best, the law promised a gain of {FC['gain_promised']:.3f}; the run delivered {FC['gain_real']:.3f}."
        cap.say(say, fade(end, el, br, lab, rel), FadeOut(tk), FadeOut(st["gl"]), FadeIn(gains), extra=T_BEAT)
        self.play(FadeOut(VGroup(ln, hd, gains, rel, br, lab, st["w"], st["ft"], st["vl"])), run_time=T_MOVE)


class Checks(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        P = canvas()
        st = first_state(P)
        for k in ("gl", "w", "ft", "vl"):
            st.pop(k)
        end, el = run_end(P)
        ft = L(f"forecast {FC['f']:.4f}").next_to(st["fd"], RIGHT, buff=0.18).shift(0.16 * UP)
        self.add(S.kicker("Why nothing warned us"), *st.values(), end, el)
        y1, y2 = 1.385, 1.362
        a1 = VGroup(
            Line(P.p(B[1]["c"], y1), P.p(TOP, y1), color=INK, stroke_width=4), Dot(P.p(TOP, y1), radius=0.07, color=INK)
        )
        t1 = stack(
            L(f"{BT['distance_third']:.1f}× out"),
            L(f"off by {BT['rmse_third']:.3f}", SMALL, INK2),
            buff=0.05,
            edge=ORIGIN,
        ).next_to(a1, UP, buff=0.1)
        a2 = VGroup(
            Line(P.p(TOP, y2), P.p(C20, y2), color=HILITE, stroke_width=4), Dot(P.p(C20, y2), radius=0.07, color=HILITE)
        )
        t2 = stack(
            L(f"{FC['distance']:.0f}× out"), L(f"off by {RUN['miss']:.3f}", SMALL, INK2), buff=0.05, edge=ORIGIN
        ).next_to(a2, UP, buff=0.1)
        cap.say(
            f"Nothing warned us. Fit two budgets, predict the third, {BT['distance_third']:.1f}× out: off by {BT['rmse_third']:.3f}.",
            FadeIn(ft),
            Create(a1),
            FadeIn(t1),
        )
        cap.say(
            f"A backtest only vouches for the distance it tested. This forecast went {FC['distance']:.0f}× out.",
            Create(a2),
            FadeIn(t2),
        )
        self.play(FadeOut(VGroup(a1, t1, a2, t2)), run_time=T_QUICK)

        wh = VGroup(*[P.whisker(b["c"], *b["l_ci"], BUD[i], 3) for i, b in enumerate(B)])
        cap.say(
            "Now drop the U's steep sides. Through the three best sizes, draw every curve with a floor that fits.",
            st["solid"].animate.set_stroke(opacity=0.25),
            FadeOut(st["dash"]),
            FadeOut(st["tags"]),
            FadeIn(wh),
        )
        xs = np.geomspace(B[0]["c"], C20, 70)
        curves = VGroup(
            *[P.curve(xs, f["E"] + f["k"] * (xs / 1e18) ** -f["g"], SKY, 2.2, o=0.8) for f in FAN["curves"]]
        )
        c = np.array(FAN["c"])
        keep = (c >= B[0]["c"]) & (c <= C20)
        fan = Polygon(
            *[P.p(x, y) for x, y in zip(c[keep], np.array(FAN["lo"])[keep])],
            *[P.p(x, y) for x, y in zip(c[keep][::-1], np.array(FAN["hi"])[keep][::-1])],
            fill_color=SKY,
            fill_opacity=0.16,
            stroke_width=0,
        )
        self.add(curves, fan)
        self.bring_to_back(fan)
        self.bring_to_back(st["gap"])
        self.bring_to_front(st["dots"], wh, st["fd"], end)
        self.play(
            LaggedStart(*[Create(m) for m in curves], lag_ratio=0.08),
            FadeIn(fan),
            *cap("They agree where there are runs, and fan out where there are none."),
            run_time=3.0,
        )
        cap.hold()
        hi = L(f"floor {FAN['floors'][1]:.2f}").next_to(P.p(C20, FAN["at20"][1]), RIGHT, buff=0.14)
        lo = L(f"floor {FAN['floors'][0]:.2f}").next_to(P.p(C20, FAN["at20"][0]), RIGHT, buff=0.14).shift(0.12 * DOWN)
        cap.say(
            f"Floors from {FAN['floors'][0]:.2f} to {FAN['floors'][1]:.2f} all fit, and the big run lands inside the fan. Three points can't pin a floor.",
            fade(hi, lo),
            extra=T_BEAT,
        )
        clear(self)
        self.spread(cap)

    def spread(self, cap):
        self.add(S.kicker("Why nothing warned us"))
        Q = Plot(
            (0, 3),
            (1.24, 1.31),
            (-5.7, -1.85, 6.4, 2.95),
            yt=[(v, f"{v:.2f}") for v in (1.25, 1.26, 1.27, 1.28, 1.29, 1.30)],
            yl="forecast for the big run, three ways",
            logx=False,
        )
        y0 = RUN["final"]
        ll = L(f"big run {y0:.4f}").next_to(Q.p(3, y0), UP, buff=0.08).align_to(Q.p(3, y0), RIGHT)
        groups = []
        for i, ((lo, hi), name) in enumerate(
            (
                (FC["ci90"], "one formula's error bar"),
                ((FO["lo"], FO["hi"]), f"four formulas that fit the {N_MOE} runs"),
                (FAN["at20"], "any floor that fits the best sizes"),
            )
        ):
            w = Q.whisker(0.5 + i, lo, hi, SKY, 7, 0.16)
            groups.append(
                VGroup(
                    w,
                    L(name).move_to(Q.p(0.5 + i, 1.24)).shift(0.3 * DOWN),
                    L(f"{lo:.3f} to {hi:.3f}", LABEL, INK2).next_to(w, RIGHT, buff=0.14),
                )
            )
        self.play(
            fade(Q, Q.hline(y0, ALLIE20, 3), ll, groups[0]),
            *cap(
                f"The law's ±{HALF:.3f} only counted the noise inside one formula. Other formulas that fit said {FO['lo']:.3f} to {FO['hi']:.3f}."
            ),
            run_time=T_MOVE,
        )
        self.wait(2.6)
        self.play(FadeIn(groups[1]), run_time=T_MOVE)
        cap.hold()
        cap.say("Only the fan reaches the result.", FadeIn(groups[2]), extra=1.4)
        clear(self)


# ------------------------------------------------------------------ why it missed


class Why(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        self.add(S.kicker("Why it missed"))
        lo, hi = CK["recipe_range"]
        rows = (("The test", f"both scored on the same {CK['moves'] / 1e6:.2f}M moves"),
                ("How parameters are counted", f"moves the forecast by {CK['count_shift']:.4f} at most"),
                ("The recipe", f"As a whole, worse at small scale, by {lo:.3f} to {hi:.3f};\nthat doesn't show its effect at {FC['distance']:.0f}×"))  # fmt: skip
        items = VGroup(L("Checked", LABEL, MUTED).move_to([-6.6, 2.75, 0], aligned_edge=LEFT))
        for i, (a, b) in enumerate(rows):
            items.add(
                stack(L(a, BODY, weight="SEMIBOLD"), L(b, LABEL, INK2, line_spacing=0.9), buff=0.1).move_to(
                    [-6.6, 1.95 - i * 1.25, 0], aligned_edge=LEFT
                )
            )
        cap.say(
            f"So why? Not the test. Not how parameters are counted. The big run's recipe was worse at small scale, which doesn't show its effect at {FC['distance']:.0f}×.",
            FadeIn(items),
        )
        cap.say("Or the law underestimates how far loss keeps falling with compute.", extra=T_BEAT)

        x, w, top, bottom = 0.9, 0.8, 2.45, -1.9
        scale = (top - bottom) / RUN["miss"]
        rt = L(f"forecast {FC['f']:.4f}").next_to([x, top, 0], UP, buff=0.1, aligned_edge=LEFT)
        title = L("One way to split the miss", LABEL, MUTED).next_to(rt, RIGHT, buff=0.6)
        rb = L(f"big run {RUN['final']:.4f}").next_to([x, bottom, 0], DOWN, buff=0.1, aligned_edge=LEFT)
        segs, y = VGroup(), top
        parts = ((CK["decay"], SKY, "last of the learning-rate decay", "read off the big run's curve", False),
                 (CK["starved"], PANEL, "token-starved runs set the floor", "tentative", True),
                 (CK["rest"], DENSE, "nothing we measured explains it", "unexplained", True))  # fmt: skip
        for v, col, a, b, warn in parts:
            h = v * scale
            r = bar(w, h - 0.04, col).move_to([x + w / 2, y - h / 2, 0])
            if col == PANEL:
                r.set_stroke(INK2, 2)
            lab = stack(
                VGroup(L(f"{v:.3f}", BODY, weight="SEMIBOLD"), S.tag(b, warn=warn)).arrange(RIGHT, buff=0.2),
                L(a, LABEL, INK2),
                buff=0.06,
            ).next_to(r, RIGHT, buff=0.22)
            segs.add(VGroup(r, lab))
            y -= h
        cap.say(
            f"One lead, part of that recipe: the big run gained {CK['decay']:.3f} below {CK['lr_end_sweep']:.0%} of peak learning rate, where the sweep's runs stop.",
            fade(rt, title, rb, segs[0]),
        )
        cap.say(
            f"One tentative: token-starved runs may pin the floor too high. The last {CK['rest']:.3f}, nothing we measured explains.",
            fade(segs[1], segs[2]),
            extra=T_BEAT,
        )
        clear(self)


# ------------------------------------------------------------------ the second forecast


def second_state(P):
    grid = np.array(RF["grid"])
    return dict(
        gap=P.gap(ANCHOR, C21),
        P=P,
        best=VGroup(
            *[
                P.hollow(b["c"], b["l_opt"], BUD4[i], 0.11, 3)
                if b["edge"]
                else P.dot(b["c"], b["l_opt"], BUD4[i], 0.11)
                for i, b in enumerate(RF["best"])
            ]
        ),
        anchors=VGroup(*[P.dot(a["c"], a["loss"], INK, 0.1) for a in RF["anchors"]]),
        curves=VGroup(*[P.curve(grid, RF["forms"][k]["curve"], SKY, 2.6, o=0.9) for k in FOUR]),
    )


class Second(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        self.add(S.kicker("The second forecast"))
        P = canvas((-5.7, -1.85, 2.9, 2.95), (1.20, 1.41), "loss on September 2026 games")
        st = second_state(P)
        budgets = [b["budget"] for b in RF["best"]]
        runs = VGroup(
            *[P.dot(r["c"], r["loss"], BUD4[budgets.index(r["budget"])], 0.04) for r in RF["runs"] if r["loss"] < 1.41]
        )
        tag = S.tag(f"each adjusted to its best learning rate, by at most {RF['max_move']:.3f}").move_to(
            P.p(XR[1], 1.385), aligned_edge=RIGHT
        )
        tag2 = S.tag("ringed: its best size is the smallest tried").next_to(tag, DOWN, buff=0.1, aligned_edge=RIGHT)
        cap.say(
            f"For Allie 2.1 we started again: {RF['n_runs']} new runs with its recipe and the big run's learning-rate decay.",
            FadeIn(P),
            LaggedStart(*[FadeIn(m) for m in sorted(runs, key=lambda m: m.get_x())], lag_ratio=0.02),
            fade(st["best"], tag, tag2),
            run_time=T_SLOW,
            extra=1.0,
        )
        al = L("two longer runs").next_to(st["anchors"], UP + RIGHT, buff=0.1)
        cap.say(
            f"Two longer runs anchor the level. They are still {C21 / ANCHOR:.0f}× short of 2.1.",
            fade(st["anchors"], al),
        )
        self.add(st["gap"])
        self.bring_to_back(st["gap"])
        vt = L("Allie 2.1").next_to(P.p(C21, 1.41), LEFT, buff=0.12).shift(0.2 * DOWN)
        self.play(
            FadeOut(tag),
            FadeOut(tag2),
            FadeIn(st["gap"]),
            LaggedStart(*[Create(c) for c in st["curves"]], lag_ratio=0.2),
            Create(P.vline(C21)),
            FadeIn(vt),
            *cap("Four formulas instead of one. Their spread is the forecast."),
            run_time=T_SLOW,
        )
        cap.hold()

        Q = Plot(
            (0, 1),
            (1.212, 1.246),
            (4.3, -1.85, 6.5, 2.95),
            yt=[(v, f"{v:.2f}") for v in (1.22, 1.23, 1.24)],
            logx=False,
        )
        qx = (Q.box[0] + Q.box[2]) / 2
        zoom = VGroup(*[Line(P.p(C21, y), [Q.box[0] - 0.6, Q.fy(y), 0], color=RULE, stroke_width=1.5) for y in Q.yr])
        band = (
            bar(Q.box[2] - Q.box[0], Q.fy(HI21) - Q.fy(LO21), SKY)
            .set_fill(opacity=0.28)
            .move_to([qx, (Q.fy(HI21) + Q.fy(LO21)) / 2, 0])
        )
        marks = VGroup(*[Q.curve((0, 1), (RF["forms"][k]["at21"],) * 2, SKY, 3) for k in FOUR])
        bl = stack(L("forecast"), L(f"{LO21:.4f} to {HI21:.4f}"), buff=0.04, edge=ORIGIN).next_to(band, UP, buff=0.1)
        rate = lr_tag().move_to(P.p(C21, 1.383) + 0.12 * LEFT, aligned_edge=RIGHT)
        cap.say(
            f"On September 2026 games: {LO21:.4f} to {HI21:.4f}. Fixed on {A21['refit_fixed']}.",
            fade(zoom, Q, band, marks, bl, rate),
        )
        nf = RF["forms"]["no_floor"]
        nm = Circle(radius=0.11, color=ROSE, stroke_width=3.5).move_to([qx, Q.fy(nf["at21"]), 0])
        rm = sorted(RF["forms"][k]["rmse"] for k in FOUR)
        t2 = S.tag(f"typical error {nf['rmse']:.4f}, against {rm[0]:.4f} to {rm[-1]:.4f}").next_to(
            rate, DOWN, buff=0.1, aligned_edge=RIGHT
        )
        cap.say(
            f"A fifth formula, with no floor, fit far worse and was left out. It said {nf['at21']:.3f}.",
            fade(nm, L(f"no floor {nf['at21']:.4f}").next_to(nm, DOWN, buff=0.08), t2),
        )
        s = Dot([qx, Q.fy(A21["sept"]), 0], radius=0.14, color=ALLIE21)
        sl = L(f"Allie 2.1 {A21['sept']:.4f}", weight="SEMIBOLD").next_to(s, DOWN, buff=0.08)
        say = f"On {A21['finished']}, the released weights scored {A21['sept']:.4f}: below the forecast again, by {LO21 - A21['sept']:.3f} to {HI21 - A21['sept']:.3f}."
        cap.say(
            say,
            FadeIn(s, scale=1.8),
            FadeIn(P.dot(C21, A21["sept"], ALLIE21, 0.13), scale=1.8),
            FadeIn(sl),
            FadeOut(t2),
            extra=T_BEAT,
        )
        ran = S.tag(
            f"as trained: forecast {LR['lo']:.4f} to {LR['hi']:.4f}, gap {GAPS[0]:.3f} to {GAPS[1]:.3f}"
        ).next_to(rate, DOWN, buff=0.1, aligned_edge=RIGHT)
        cap.say(
            f"That forecast took the law's best learning rate. For the run as trained, the law says {LR['lo']:.4f} to {LR['hi']:.4f}: a gap of {GAPS[0]:.3f} to {GAPS[1]:.3f}.",
            FadeIn(ran),
            extra=T_BEAT,
        )
        fls = [RF["forms"][k]["floor"] for k in FOUR]
        cap.say(
            f"These floors sit at {min(fls):.2f} to {max(fls):.2f}; the first law's at {LAW['E']:.2f}, on another test. A floor is fit, not measured.",
            FadeOut(rate),
            FadeOut(ran),
            extra=T_BEAT,
        )
        clear(self)


class HowBig(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        self.add(S.kicker("The other question"))
        ox, oy = -5.85, -1.85
        sx, sy = 7.2 / HB["first_n"], 4.6 / HB["refit_d"][1]

        def rect(n, d):
            return Rectangle(width=n * sx, height=d * sy, stroke_color=SKY, stroke_width=3.5).move_to(
                [ox, oy, 0], aligned_edge=DOWN + LEFT
            )

        ax = VGroup(
            Line([ox, oy, 0], [ox + 7.5, oy, 0], color=RULE, stroke_width=2.5),
            Line([ox, oy, 0], [ox, oy + 4.85, 0], color=RULE, stroke_width=2.5),
        )
        xn = L("parameters used per move", SMALL, INK2).move_to([ox + 3.7, oy - 0.3, 0])
        yn = L("training tokens", SMALL, INK2).move_to([-6.6, oy + 5.1, 0], aligned_edge=LEFT)
        first = rect(HB["first_n"], HB["first_d"])
        fl = L(f"first law: {HB['first_n'] / 1e9:.1f}B parameters, {HB['first_d'] / 1e9:.0f}B tokens").next_to(
            first, RIGHT, buff=0.2, aligned_edge=DOWN
        )
        n0, n1, d0, d1 = *HB["refit_n"], *HB["refit_d"]
        refit = VGroup(rect(n1, d0), rect(n0, d1))
        rl = (
            L(f"new law: {n0 / 1e9:.1f} to {n1 / 1e9:.1f}B parameters, {d0 / 1e9:.0f} to {d1 / 1e9:.0f}B tokens")
            .next_to(refit, RIGHT, buff=0.2)
            .align_to(refit, UP)
        )
        a = bar(A21["n"] * sx, A21["d"] * sy, ALLIE21).move_to([ox, oy, 0], aligned_edge=DOWN + LEFT)
        al = (
            L(f"Allie 2.1: {A21['n'] / 1e9:.2f}B parameters, {A21['d'] / 1e9:.0f}B tokens")
            .next_to(a, RIGHT, buff=0.2)
            .align_to(a, UP)
        )
        note = T(f"same area: {sci(C21)} FLOPs", SMALL, INK2).move_to([6.6, 2.95, 0], aligned_edge=RIGHT)
        self.play(
            fade(ax, xn, yn, first, fl, note),
            *cap(
                "How big? At 2.1's compute, the first law wanted a huge model on few tokens; the new law, a far smaller one on more."
            ),
            run_time=T_MOVE,
        )
        self.wait(3.2)
        self.play(fade(refit, rl), run_time=T_MOVE)
        cap.hold()
        self.add(a)
        self.bring_to_back(a)
        cap.say(
            f"Allie 2.1 ran between them. By the new law, loss barely moves near the best size: under {HB['flat']:.3f}.",
            fade(a, al),
            extra=T_BEAT,
        )
        clear(self)


class Lessons(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        self.add(S.kicker("What it taught us"))
        p1 = canvas((-5.9, -1.65, -0.55, 2.0), (1.22, 1.41), yl=None, xl=None)
        p2 = canvas((1.0, -1.65, 6.35, 2.0), (1.20, 1.39), yl=None, xl=None)
        h1 = stack(
            L("First forecast: Allie 2.0's big run", weight="SEMIBOLD"),
            L("law on the first sweep; loss on the main test set, July games", SMALL, INK2),
            buff=0.05,
        ).move_to([-6.5, 2.65, 0], aligned_edge=LEFT)
        h2 = stack(
            L("Second forecast: Allie 2.1", weight="SEMIBOLD"),
            L("law on a new sweep with its recipe; loss on September games", SMALL, INK2),
            buff=0.05,
        ).move_to([0.45, 2.65, 0], aligned_edge=LEFT)
        solid, dash = law_line(p1)
        dots1 = VGroup(*[p1.dot(b["c"], b["l_opt"], BUD[i], 0.08) for i, b in enumerate(B)])
        v1 = VGroup(
            p1.gap(TOP, C20),
            p1,
            solid,
            dash,
            dots1,
            p1.hollow(C20, FC["f"], SKY, 0.1, 3.5),
            p1.dot(C20, RUN["final"], ALLIE20, 0.12),
        )
        r1 = L(f"{FC['distance']:.0f}× out", LABEL, INK2).move_to(p1.p(np.sqrt(TOP * C20), 1.39))
        l1 = (
            stack(L(f"forecast {FC['f']:.4f}"), L(f"scored {RUN['final']:.4f}"), buff=0.06)
            .move_to(p1.p(C20, 1.35), aligned_edge=RIGHT)
            .shift(0.12 * LEFT)
        )
        bx = p1.fx(C20) + 0.3
        b1 = bracket(bx, p1.fy(FC["f"] - 0.003), p1.fy(RUN["final"] + 0.003))
        g1 = L(f"{RUN['miss']:.3f}", LABEL, HILITE, weight="SEMIBOLD").next_to(b1, RIGHT, buff=0.08)
        st = second_state(p2)
        v2 = VGroup(
            st["gap"],
            p2,
            st["curves"],
            st["best"],
            st["anchors"],
            p2.whisker(C21, LO21, HI21, SKY, 5),
            p2.dot(C21, A21["sept"], ALLIE21, 0.12),
        )
        r2 = L(f"{C21 / ANCHOR:.0f}× out", LABEL, INK2).move_to(p2.p(np.sqrt(ANCHOR * C21), 1.37))
        l2 = stack(
            L(f"forecast {LO21:.4f}–{HI21:.4f}"),
            L(f"scored {A21['sept']:.4f}"),
            L(f"as trained {LR['lo']:.4f}–{LR['hi']:.4f}, gap {GAPS[0]:.3f}–{GAPS[1]:.3f}", SMALL, INK2),
            buff=0.06,
        ).move_to(p2.p(C21, 1.33), aligned_edge=RIGHT)
        g2 = L(f"{LO21 - A21['sept']:.3f} to {HI21 - A21['sept']:.3f}", LABEL, HILITE, weight="SEMIBOLD").next_to(
            p2.p(C21, A21["sept"]), DOWN + LEFT, buff=0.1
        )
        group = VGroup(v1, h1, r1, l1, b1, g1, v2, h2, r2, l2, g2)
        rate = lr_tag().move_to([0, -2.32, 0])
        cap.say(
            "Side by side, at one scale: a shorter reach, a smaller gap. Both landed below.",
            FadeIn(group),
            FadeIn(rate),
            extra=1.0,
        )
        o5 = A21["oct5"]
        note = S.tag(
            f"Also: on {o5['date']}, before 2.1's recipe changed, built up from 2.0's big run: {o5['lo']:.3f} to {o5['hi']:.3f} on the main test. Final weights: {A21['final_main']:.4f}.",
            warn=True,
        ).move_to([0, -2.32, 0])
        cap.say(
            "These gaps aren't like for like: other laws, other test games. The recipe changed between them, so the smaller gap isn't all the law's doing.",
            FadeOut(rate),
            FadeIn(note),
            extra=T_BEAT,
        )
        self.play(FadeOut(group), FadeOut(note), run_time=T_QUICK)
        items = VGroup()
        for a, b in (("A good fit is not a good forecast.", "A backtest only vouches for the distance it tested."),
                     ("Anchor on the biggest real run.", "And train small runs the way the big one will train."),
                     ("Report the spread across formulas.", "One formula's error bar only counts its own noise.")):  # fmt: skip
            items.add(stack(L(a, TITLE, font=SLAB, weight="SEMIBOLD"), L(b, S.CAPTION, INK2), buff=0.14))
        items.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=MARGIN).shift(0.45 * UP)
        self.play(LaggedStart(*[FadeIn(i) for i in items], lag_ratio=0.55), *cap.clear(), run_time=2.4)
        self.wait(3.4)
        cap.say(
            "Twice now, loss kept falling further than a law fit on smaller runs said. Two runs aren't a rule.",
            extra=1.5,
        )
        clear(self, T_SLOW)
