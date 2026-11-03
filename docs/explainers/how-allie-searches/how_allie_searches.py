"""How Allie Searches: why a model that predicts humans has to calculate to play like them, what the bot's rule does on
a real position, what runs during a live move and how close it gets. Every number comes from data.json (measure.py)."""

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
    Arrow,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Group,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    Scene,
    Square,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S  # noqa: E402

D = json.loads((HERE / "data.json").read_text())
P, G, Z = D["position"], D["aggregates"], D["sources"]
FMTS = ("bullet", "blitz", "rapid", "classical")
BINS = list(range(800, 2601, 200))
RUNGS = Z["rule"]["ladder"]
MOVE = {m["san"]: m for m in P["moves"]}
BY_RATING = {r["rating"]: r for r in P["by_rating"]}
ROAD = G["roads"]["classical_2400"]
VIEW = int(Z["rule"]["view"].split("/")[0][1:])
LEFT_EDGE = -S.FRAME_W / 2 + S.MARGIN
RIGHT_EDGE = S.FRAME_W / 2 - S.MARGIN
VIDEO = dict(
    name="how-allie-searches",
    scenes=["Title", "Position", "Target", "BlindSpot", "Search", "Tilt", "Live", "Close", "Short", "End"],
    gif=(["Search", 12.0], 26),
    poster=["Title", 5.0],
)


def cell(f, b):
    return Z["cells"][f"{f}/{b}"]


def minus(s):
    return str(s).replace("-", "−")


def signed(x, d=2):
    return minus(f"{x:+.{d}f}")


def pct(x, d=None):
    x = 100 * x
    d = d if d is not None else (0 if x >= 10 else 1)
    return f"{x:.{d}f}%"


def score(v):
    """White's expected score from a searched value, v = arctanh(0.98 (W - L))."""
    return (1 + np.tanh(v) / 0.98) / 2


def rect(w, h, color=S.PANEL):
    return Rectangle(width=max(w, 0.004), height=h, fill_color=color, fill_opacity=1, stroke_width=0)


def outline(w, h, color=S.HILITE, stroke=4):
    return Rectangle(width=w, height=h, fill_opacity=0, stroke_width=stroke, stroke_color=color)


def hbar(x, y, w, h=0.34, color=S.BRASS):
    return rect(w, h, color).move_to([x, y, 0], aligned_edge=LEFT)


def at(s, x, y, px=S.LABEL, color=S.INK2, edge=LEFT, **kw):
    """Text with its line at y whatever its descenders: y is the middle of a capital, not of the ink."""
    t = S.text(s, px, color, **kw).move_to([x, y, 0], aligned_edge=edge)
    if not {*s} & {*"gjpqy,;()[]βγηρ/|"} or "\n" in s:
        return t
    return t.shift(DOWN * (S.text("Hp", px).height - S.text("H", px).height) / 2)


def chip(color, s, px=S.LABEL):
    return VGroup(rect(0.2, 0.2, color), S.text(s, px, S.INK)).arrange(RIGHT, buff=0.1)


def fade(*mobjects):
    return [FadeIn(m) for m in mobjects]


def wipe(scene, cap=None, keep=()):
    """Fade out everything on screen but `keep`, the caption included."""
    gone = [m for m in scene.mobjects if all(m is not h for h in keep)]
    return [FadeOut(Group(*gone)), *(cap.clear() if cap else [])]


def cells_grid(cw, ch, x0, y0, tile):
    """Time control by rating: its row labels, column labels and tile(f, b, box) per cell."""
    pos = lambda r, i: [x0 + i * (cw + 0.05), y0 - r * (ch + 0.05), 0]  # noqa: E731
    rows = VGroup(
        *[at(f.capitalize(), x0 - cw / 2 - 0.12, pos(r, 0)[1], S.LABEL, S.INK2, RIGHT) for r, f in enumerate(FMTS)]
    )
    cols = VGroup(*[at(str(b), pos(0, i)[0], y0 + ch / 2 + 0.22, S.SMALL, S.MUTED, ORIGIN) for i, b in enumerate(BINS)])
    tiles = VGroup(
        *[tile(f, b, rect(cw, ch).move_to(pos(r, i))) for r, f in enumerate(FMTS) for i, b in enumerate(BINS)]
    )
    return VGroup(rows, cols), tiles


class Plot:
    """Data to frame coordinates inside box = (left, right, bottom, top)."""

    def __init__(self, box, xr, yr, ylog=False, xlog=False):
        self.box, self.xr, self.yr, self.ylog, self.xlog = box, xr, yr, ylog, xlog

    def x(self, v):
        f = np.log if self.xlog else float
        return self.box[0] + (f(v) - f(self.xr[0])) / (f(self.xr[1]) - f(self.xr[0])) * (self.box[1] - self.box[0])

    def y(self, v):
        f = np.log if self.ylog else float
        return self.box[2] + (f(v) - f(self.yr[0])) / (f(self.yr[1]) - f(self.yr[0])) * (self.box[3] - self.box[2])

    def p(self, x, y):
        return np.array([self.x(x), self.y(y), 0.0])

    def grid(self, ys, xs=(), yfmt=str, xfmt=str, left=True, px=S.SMALL):
        out = VGroup()
        for v in ys:
            out.add(Line(self.p(self.xr[0], v), self.p(self.xr[1], v), stroke_color=S.RULE, stroke_width=1.5))
            if left:
                out.add(at(yfmt(v), self.box[0] - 0.12, self.y(v), px, S.MUTED, RIGHT))
        for v in xs:
            out.add(at(xfmt(v), self.x(v), self.box[2] - 0.26, px, S.MUTED, ORIGIN))
        return out

    def line(self, xs, ys, color=S.INK, width=4):
        return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(
            [self.p(x, y) for x, y in zip(xs, ys)]
        )

    def band(self, xs, lo, hi, color=S.INK, opacity=0.18):
        pts = [self.p(x, y) for x, y in zip(xs, hi)] + [self.p(x, y) for x, y in zip(xs[::-1], lo[::-1])]
        return Polygon(*pts, fill_color=color, fill_opacity=opacity, stroke_width=0)


def segments(a, b):
    """Straight strokes a[i] -> b[i] as a VMobject's cubic points."""
    a, b = np.asarray(a), np.asarray(b)
    out = np.zeros((4 * len(a), 3))
    out[0::4], out[1::4], out[2::4], out[3::4] = a, a + (b - a) / 3, a + 2 * (b - a) / 3, b
    return out


class Growth(VGroup):
    """The real search tree, drawn up to some number of leaves. Radial: root moves clockwise from the top by falling
    prior, a subtree's angle its share of the final leaves, depth outward, zoomed to the depth reached so far. An edge
    wears its node's own read: blue where White is winning, red where White is losing."""

    BANDS = (-0.6, -0.3, -0.1, 0.1, 0.3, 0.6)

    def __init__(self, center=(0, 0), radius=3.0):
        super().__init__()
        t = P["tree"]
        self.parent, self.cost, self.level = np.array(t["parent"]), np.array(t["cost"]), np.array(t["depth"])
        n, prior = len(self.parent), t["prior"]
        kids = [[] for _ in range(n)]
        for c in range(1, n):
            kids[self.parent[c]].append(c)
        weight = np.ones(n)
        for c in range(n - 1, -1, -1):
            weight[c] = max(1.0, sum(weight[k] for k in kids[c]))
        lo, hi = np.zeros(n), np.zeros(n)
        hi[0] = 2 * np.pi
        for c in range(n):
            start = lo[c]
            for k in sorted(kids[c], key=lambda j: -prior[j]):
                lo[k], hi[k] = start, start + (hi[c] - lo[c]) * weight[k] / max(weight[c], 1.0)
                start = hi[k]
        self.lo, self.hi = lo, hi
        self.center_, self.radius = np.array([*center, 0.0]), radius
        mid = (lo + hi) / 2
        self.unit = np.c_[np.sin(mid), np.cos(mid), np.zeros(n)]
        self.band = np.digitize(np.array(t["white_q"]), self.BANDS)
        shades = (
            [S.mix(S.DIV_MID, S.DIV_HIGH, v) for v in (1, 0.8, 0.45)]
            + [S.DIV_MID]
            + [S.mix(S.DIV_MID, S.DIV_LOW, v) for v in (0.45, 0.8, 1)]
        )
        self.strokes = [VMobject(stroke_color=c, fill_opacity=0) for c in shades]
        self.tips = [VMobject(stroke_color=c, fill_opacity=0) for c in shades]
        self.add(*self.strokes, *self.tips)
        # the depth to zoom to, smoothed over a doubling of leaves either way
        self.grid = np.arange(-1, 10.01, 0.05)
        deepest = np.array([self.level[self.cost <= 2**g].max() for g in self.grid], float)
        self.zoom = np.convolve(np.pad(deepest, 20, mode="edge"), np.ones(41) / 41, mode="valid") + 0.5
        self.xy = np.tile(self.center_, (n, 1))
        self.show(0)

    def show(self, budget):
        z = max(np.interp(np.log2(max(budget, 0.5)), self.grid, self.zoom), 1.5)
        self.xy = self.center_ + self.unit * (self.radius * (self.level / z) ** 0.55)[:, None]
        live = (self.cost <= budget) & (self.parent >= 0)
        near = np.clip((10 - z) / 8, 0, 1)
        off = np.array([0.02 + 0.03 * near, 0, 0])
        for i, (m, tip) in enumerate(zip(self.strokes, self.tips)):
            c = np.flatnonzero(live & (self.band == i))
            m.set_stroke(width=1.5 + 2.5 * near)
            tip.set_stroke(width=120 * off[0])
            m.points = segments(self.xy[self.parent[c]], self.xy[c]) if len(c) else np.zeros((0, 3))
            tip.points = segments(self.xy[c] - off, self.xy[c] + off) if len(c) else np.zeros((0, 3))
        return self

    def node(self, line):
        return next(int(k) for k, v in P["tree"]["lines"].items() if v == line)

    def path(self, line, budget, width=5):
        chain, n = [], self.node(line)
        while n:
            if self.cost[n] <= budget:
                chain.append(self.xy[n])
            n = self.parent[n]
        chain.append(self.xy[0])
        return VMobject(stroke_color=S.HILITE, stroke_width=width).set_points_as_corners(chain[::-1])


def arrow(board, a, b, color=S.HILITE, width=7):
    return Arrow(
        board.at(a),
        board.at(b),
        buff=0.04,
        color=color,
        stroke_width=width,
        tip_length=0.2,
        max_tip_length_to_length_ratio=0.6,
        max_stroke_width_to_length_ratio=20,
    ).set_z_index(4)


def tint(board, name, color=S.HILITE, opacity=0.55):
    s = board.squares[name]
    return Square(s.width, stroke_width=0, fill_color=color, fill_opacity=opacity).move_to(s).set_z_index(1)


def demo_board(size=5.0, center=(-3.95, 0.25)):
    b = S.board(P["fen"], size=size)
    for p in b.pieces.values():
        p.set_z_index(3)
    return b.move_to([*center, 0])


YR = (-0.08, 0.3)
ROAD_PLOT = Plot((-5.2, 5.6, -1.85, 2.55), (5.0, 1.0), YR)


def road_frame():
    pl = ROAD_PLOT
    out = pl.grid(
        (-0.05, 0, 0.1, 0.2, 0.3),
        (5, 4, 3, 2, 1),
        yfmt=lambda v: signed(v) if v else "0",
        xfmt=lambda v: f"{v}%",
        px=S.LABEL,
    )
    out.add(Line(pl.p(5.0, 0), pl.p(1.0, 0), stroke_color=S.MUTED, stroke_width=3))
    out.add(at("Blunder rate, stronger to the right", 0.2, -2.5, S.LABEL, S.INK2, ORIGIN))
    out.add(
        at(
            "Surprise at the human's move, against Allie alone, in nats. Classical, rated 2400",
            LEFT_EDGE,
            2.95,
            S.LABEL,
        )
    )
    hx = ROAD["human"]["blunder"]
    out.add(Line(pl.p(hx, YR[0]), pl.p(hx, YR[1]), stroke_color=S.INK, stroke_width=4))
    out.add(at(f"Humans: {hx:.1f}%", pl.x(hx) + 0.14, pl.y(YR[1]) - 0.2, S.LABEL, S.INK, weight="SEMIBOLD"))
    origin = rect(0.2, 0.2, S.DENSE).move_to(pl.p(ROAD["raw"]["blunder"], 0))
    return out, VGroup(origin, S.text("Allie alone", S.LABEL).next_to(origin, UP, buff=0.14))


def sharpened():
    rows = [t for t in ROAD["temperature"] if t["ce"] <= YR[1]]
    pts = [(ROAD["raw"]["blunder"], 0.0)] + [(t["blunder"], t["ce"]) for t in rows]
    note = f"Sharpened: temperature {rows[-1]['t']}, {signed(rows[-1]['ce'])} nats"
    return ROAD_PLOT.line(*zip(*pts), S.ROSE, 5), S.text(note, S.LABEL, S.INK).next_to(
        ROAD_PLOT.p(*pts[-1]), LEFT, buff=0.15
    )


class Title(Scene):
    def construct(self):
        S.ground(self)
        name = VGroup(S.title("How Allie", S.DISPLAY), S.title("Searches", S.DISPLAY, S.HILITE)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.1
        )
        name.move_to([LEFT_EDGE, 1.0, 0], aligned_edge=LEFT)
        sub = S.text("Predicting a strong player\nis not playing like one", S.HEAD, S.INK2, line_spacing=0.9)
        sub.next_to(name, DOWN, aligned_edge=LEFT, buff=0.5)
        foot = at("Allie 2.1 · github.com/y0mingzhang/allie", LEFT_EDGE, -3.4, S.SMALL, S.MUTED)
        tree = Growth(center=(3.6, 0.0), radius=3.4)
        e = ValueTracker(0.0)
        tree.add_updater(lambda m: m.show(2 ** e.get_value()))
        self.add(tree)
        self.play(FadeIn(name), FadeIn(sub), FadeIn(foot), e.animate(run_time=4.0, rate_func=linear).set_value(10.0))
        tree.clear_updaters()
        self.wait(1.5)
        self.play(FadeOut(VGroup(name, sub, foot, tree)), run_time=S.T_MOVE)


class Position(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("One position")
        b = demo_board()
        x0 = -0.6
        head = at("White to move", x0, 2.72, S.HEAD, S.INK, weight="SEMIBOLD")
        who = at(
            f"White {P['white']}, Black {P['black']} · rapid, {P['time_control'][0] // 60} minutes · September 2026",
            x0,
            2.16,
            S.LABEL,
        )
        cap.say(
            f"Allie predicts the move a human of a given rating will play. Here, in a real game, White is rated {P['white']}.",
            *fade(k, b, head, who),
        )
        threats = VGroup(arrow(b, "e7", "d6", S.FLAG), arrow(b, "c8", "c6", S.FLAG), arrow(b, "b7", "c6", S.FLAG))
        cap.say("White is a piece up, but both its pieces are attacked.", *[Create(t) for t in threats])
        p = MOVE["Rd7+"]["p"], MOVE["Bxb7"]["p"]
        rows = VGroup()
        for i, (m, v) in enumerate((("Rd7+", p[0]), ("Bxb7", p[1]), (f"{len(P['moves']) - 2} others", 1 - sum(p)))):
            y = 0.95 - 0.68 * i
            bar = hbar(x0 + 2.05, y, 4.0 * v, 0.44, S.DENSE)
            rows.add(
                VGroup(at(m, x0, y, S.BODY, S.INK, font=S.MONO), bar, at(pct(v), x0 + 2.19 + 4.0 * v, y, S.BODY, S.INK))
            )
        title = at("Allie's prediction for this player", x0, 1.58)
        cap.say(
            "Allie expects Rd7+ about two times in three, Bxb7 about one in three.",
            FadeOut(threats),
            *fade(title, rows),
        )
        bishop, pawn, rook, king, wking, wrook = (b.pieces[s] for s in ("c6", "b7", "c8", "e7", "f2", "d6"))
        take = arrow(b, "c6", "b7")
        cap.say(
            "Bxb7 takes a pawn and hits the rook. If Kxd6, then Bxc8.",
            rows[1][1].animate.set_fill(S.HILITE),
            Create(take),
        )
        fish = VGroup(
            S.text("Stockfish, for White", S.LABEL),
            S.text(f"Rd7+  {signed(MOVE['Rd7+']['stockfish'] / 100, 1)}", S.BODY, S.INK, font=S.MONO),
            S.text(f"Bxb7  {signed(MOVE['Bxb7']['stockfish'] / 100, 1)}", S.BODY, S.HILITE, font=S.MONO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        fish.move_to([x0, -1.42, 0], aligned_edge=LEFT)
        note = S.text(
            "Stockfish only scores moves here. No engine\nevaluation ever trained Allie, though a small\nshare of its training games are engine games.",
            S.LABEL,
            S.MUTED,
            line_spacing=0.9,
        ).move_to([x0 + 3.35, -1.42, 0], aligned_edge=LEFT)
        check = tint(b, "f2", S.FLAG)
        self.play(
            FadeOut(take),
            FadeOut(pawn),
            bishop.animate.move_to(b.at("b7")),
            *cap("But Black checks first: Rc2+. Then Kxd6 wins the rook for nothing."),
            run_time=S.T_MOVE,
        )
        self.play(rook.animate.move_to(b.at("c2")), FadeIn(check), run_time=S.T_MOVE)
        self.play(FadeOut(check), wking.animate.move_to(b.at("e3")), run_time=S.T_MOVE)
        self.play(FadeOut(wrook), king.animate.move_to(b.at("d6")), FadeIn(fish), FadeIn(note), run_time=S.T_MOVE)
        cap.hold(0.8)
        fresh = demo_board()
        played = fresh.pieces["d6"]
        self.play(
            FadeOut(VGroup(bishop, rook, king, wking)),
            FadeIn(VGroup(*fresh.pieces.values())),
            *cap(f"A strong player calculates this. This one played {P['human']}."),
            run_time=S.T_MOVE,
        )
        self.play(played.animate.move_to(b.at("d7")), run_time=S.T_MOVE)
        cap.hold()
        cap.say("Rolling the dice on Allie's prediction would blunder here about one time in three.", extra=0.3)
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class Target(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("The target")
        c = cell("classical", 2400)
        (ha, hb), (ra, rb) = c["human"], c["raw"]
        gaps = [abs(v) for reads in Z["raw_read"].values() for v in reads]
        lower = f"{round(min(gaps), -2)} to {round(max(gaps), -2)}"
        xs = (LEFT_EDGE, -1.4, 2.9)
        big = lambda s, x, y, color=S.INK: at(s, x, y, S.TITLE, color, weight="SEMIBOLD")  # noqa: E731
        table = VGroup(
            at("Humans rated 2400-2599", xs[1], 2.3, S.BODY),
            at("Allie alone, same ratings", xs[2], 2.3, S.BODY),
            Line([xs[0], 1.92, 0], [RIGHT_EDGE, 1.92, 0], stroke_color=S.RULE, stroke_width=2),
            at("Move accuracy", xs[0], 1.25, S.BODY),
            big(f"{ha:.1f}%", xs[1], 1.25),
            big(f"{ra:.1f}%", xs[2], 1.25),
            at("Blunder rate", xs[0], 0.05, S.BODY),
            big(f"{hb:.1f}%", xs[1], 0.05),
            big(f"{rb:.1f}%", xs[2], 0.05, S.HILITE),
            at(f"plays like humans rated {lower}\npoints lower, on these two numbers", xs[2], -0.82, S.BODY, S.HILITE, line_spacing=0.9),
            VGroup(
                S.text(f"Classical games, September 2026, {c['n']:,} positions. Stockfish at depth {Z['stockfish_depth']} scores every legal move.", S.SMALL, S.MUTED),
                S.text(f"Accuracy: Lichess's formula on the win chance a move keeps. Blunder: it loses {Z['blunder_points']} or more points of win chance.", S.SMALL, S.MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([xs[0], -1.9, 0], aligned_edge=LEFT),
        )  # fmt: skip
        cap.say(
            f"No fluke. Rolling those dice, call it Allie alone, blunders {rb / hb:.1f} times as often as real 2400s: it plays like humans rated {lower} points lower, on these two numbers.",
            *fade(k, table),
            extra=0.3,
        )

        def tile(f, b, box):
            a, bl = cell(f, b)["human"]
            return VGroup(
                box,
                VGroup(S.text(f"{a:.1f}", S.LABEL, S.INK), S.text(f"{bl:.1f}", S.LABEL, S.HILITE))
                .arrange(DOWN, buff=0.06)
                .move_to(box),
            )

        x0 = -5.05
        labels, tiles = cells_grid(1.13, 0.86, x0, 1.95, tile)
        key = (
            VGroup(
                S.text("Humans' move accuracy, %", S.SMALL, S.INK),
                S.text("Humans' blunder rate, %", S.SMALL, S.HILITE),
                S.text("Ratings in bins of 200 points", S.SMALL, S.MUTED),
            )
            .arrange(RIGHT, buff=0.6)
            .move_to([x0 - 1.13 / 2, -2.28, 0], aligned_edge=LEFT)
        )
        self.play(
            FadeOut(table),
            *cap(
                f"The goal: match those two numbers at every time control and rating. {len(Z['cells'])} cells, {Z['tests']['of']} tests."
            ),
            run_time=S.T_MOVE,
        )
        self.play(FadeIn(labels), LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.012), FadeIn(key), run_time=1.6)
        cap.hold()
        cap.say(
            "No stronger, no weaker. And human moves must not get measurably less likely than under Allie alone.",
            extra=0.3,
        )
        frame, origin = road_frame()
        path, end = sharpened()
        self.play(FadeOut(VGroup(labels, tiles, key)), run_time=S.T_QUICK)
        self.play(
            *fade(frame, origin),
            *cap("The easy fix: sharpen the prediction. Blunders fall, but human moves get far more surprising."),
            run_time=S.T_MOVE,
        )
        self.play(Create(path), FadeIn(end), run_time=1.6)
        cap.hold()
        limit = ROAD["argmax"]["blunder"]
        stop = DashedLine(ROAD_PLOT.p(limit, YR[0]), ROAD_PLOT.p(limit, YR[1]), stroke_color=S.ROSE, stroke_width=3)
        said = S.text(f"Always its top choice: {limit:.1f}%", S.LABEL, S.INK).next_to(
            ROAD_PLOT.p(limit, -0.035), LEFT, buff=0.14
        )
        cap.say(
            f"Even always playing its top choice stops at {limit:.1f}% blunders. Sharpening adds no information.",
            Create(stop),
            FadeIn(said),
            extra=0.3,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class BlindSpot(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        B = G["blind_spot"]
        k = S.kicker("What is missing")
        title = at(
            f"Moves Allie gives at least {pct(B['threshold'])}: how often humans play them ÷ how often Allie predicts",
            LEFT_EDGE,
            2.85,
            S.BODY,
            S.INK,
        )
        w, gap, x0 = 2.9, 0.3, -5.85
        plots = [
            Plot((x0 + i * (w + gap), x0 + i * (w + gap) + w, -1.8, 1.95), (800, 2600), (0, 1.25)) for i in range(4)
        ]
        frames = VGroup(at("Rating of the player to move", 0, -2.47, S.SMALL, S.MUTED, ORIGIN))
        for i, (pl, f) in enumerate(zip(plots, FMTS)):
            frames.add(pl.grid((0, 0.5, 1.0), (1000, 1800, 2600), yfmt=lambda v: f"{v:.1f}", left=i == 0))
            frames.add(at(f.capitalize(), (pl.box[0] + pl.box[1]) / 2, 2.25, S.BODY, S.INK2, ORIGIN, weight="SEMIBOLD"))
        fine = VGroup(*[pl.line(BINS, B[f]["fine"], S.INK, 4).set_z_index(2) for pl, f in zip(plots, FMTS)])
        fine_note = at("Stockfish: fine", plots[0].x(1700), plots[0].y(1.2), S.LABEL, S.INK, ORIGIN)
        lines, bands = VGroup(), VGroup()
        for pl, f in zip(plots, FMTS):
            bands.add(pl.band(BINS, B[f]["lo"], B[f]["hi"], S.DIV_HIGH, 0.3))
            lines.add(pl.line(BINS, B[f]["blunder"], S.DIV_HIGH, 5))
        bad_note = chip(S.DIV_HIGH, "Stockfish: blunder").move_to(
            [plots[0].x(800), plots[0].y(0.62), 0], aligned_edge=LEFT
        )
        cap.say(
            f"What is missing? Moves Allie gives at least {pct(B['threshold'])} that Stockfish calls fine get played as often as predicted.",
            *fade(k, title, frames),
            Create(fine),
            FadeIn(fine_note),
            run_time=1.2,
        )
        cap.say(
            "Those it calls blunders: below 1800, also about as predicted. Allie's numbers are honest.",
            FadeIn(bad_note),
            LaggedStart(*[FadeIn(b) for b in bands], lag_ratio=0.3),
            LaggedStart(*[Create(m) for m in lines], lag_ratio=0.3),
            run_time=2.0,
        )
        c24, i24 = B["classical"], BINS.index(2400)
        z, t = c24["blunder"][i24], c24["tilted"][i24]
        big = lambda v: S.text(f"{v:.2f}", S.BODY, S.HILITE, weight="SEMIBOLD")  # noqa: E731
        mark = big(z).next_to(plots[3].p(2400, z), LEFT, buff=0.38).shift(DOWN * 0.15)
        cap.say(
            f"But strong players avoid them: {z:.2f} [{c24['lo'][i24]:.2f}, {c24['hi'][i24]:.2f}] at classical 2400, {c24['hits'][i24]} plays against {c24['predicted'][i24]:.0f} predicted. Bullet stays near 1.",
            FadeIn(mark),
        )
        cap.say(
            f"At strong, slow play Allie gives these moves too much weight: about {1 / z:.1f} times at classical 2400."
        )
        with_search = VGroup(*[pl.line(BINS, B[f]["tilted"], S.BRASS, 4).set_z_index(1) for pl, f in zip(plots, FMTS)])
        searched = chip(S.BRASS, "Stockfish: blunder, with search").move_to(
            [plots[0].x(800), plots[0].y(0.42), 0], aligned_edge=LEFT
        )
        cases = ", ".join(
            f"{f} {r} {B[f]['blunder'][BINS.index(r)]:.2f} to {B[f]['tilted'][BINS.index(r)]:.2f}"
            for f, r in (("classical", 2400), ("rapid", 2600), ("classical", 2200))
        )
        cap.say(
            f"Search corrects most of it: classical 2400 goes from {z:.2f} to {t:.2f}.",
            Create(with_search),
            FadeIn(searched),
            FadeIn(big(t).next_to(plots[3].p(2400, t), DOWN, buff=0.12)),
            FadeIn(S.tag(f"With search: {cases}").move_to([RIGHT_EDGE, S.KICKER_Y, 0], aligned_edge=RIGHT)),
            run_time=1.2,
            extra=0.8,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class Search(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("The search")
        b = demo_board(3.1, (-4.95, 1.42))
        tree = Growth(center=(2.75, 0.22), radius=2.95)
        e = ValueTracker(-1.0)
        tree.add_updater(lambda m: m.show(2 ** e.get_value()))
        root = rect(0.13, 0.13, S.INK).move_to(tree.center_)
        count = always_redraw(
            lambda: at(
                f"{int(min(2 ** e.get_value(), 1024)):,}" if e.get_value() >= 0 else "0",
                RIGHT_EDGE,
                2.85,
                S.TITLE,
                S.INK,
                RIGHT,
                weight="SEMIBOLD",
            )
        )
        unit = at("leaves", RIGHT_EDGE, 2.3, S.LABEL, S.INK2, RIGHT)
        key = VGroup(chip(S.DIV_LOW, "White winning"), chip(S.DIV_HIGH, "White losing")).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12
        )
        key.move_to([-3.0, 2.62, 0], aligned_edge=LEFT)
        head = at("White's expected score", LEFT_EDGE, -0.62)
        xb, wb = -5.4, 2.7
        ys = {"Rd7+": -1.15, "Bxb7": -1.8}
        names = VGroup(*[at(m, LEFT_EDGE, y, S.BODY, S.INK, font=S.MONO) for m, y in ys.items()])
        steps, shown = np.array(P["leaf_steps"]), ValueTracker(0.0)

        def bars():
            out = VGroup()
            for m, y in ys.items():
                v = score(np.interp(2 ** e.get_value(), steps, P["values_so_far"][m]))
                out.add(hbar(xb, y, wb, 0.38, S.PANEL), hbar(xb, y, wb * v, 0.38, S.DIV_LOW if v > 0.5 else S.DIV_HIGH))
                out.add(at(pct(v, 0), xb + wb + 0.14, y, S.BODY, S.INK))
            return out.set_opacity(shown.get_value())

        cur = always_redraw(bars)

        def grow(to, seconds):
            return e.animate(run_time=seconds, rate_func=linear).set_value(np.log2(to))

        def named(line, s):
            """A move's label, appearing when its node is read and riding it as the tree zooms out."""
            n = tree.node(line)

            def ride(m):
                m.next_to(tree.xy[n], RIGHT, buff=0.1)
                m.set_opacity(np.clip((e.get_value() - np.log2(tree.cost[n])) / 0.25, 0, 1))

            return S.text(s, S.LABEL, S.INK, font=S.MONO).add_updater(ride, call_updater=True)

        self.add(tree, cur)
        kl = S.tag("KL search, after piKL (Jacob et al., 2022)").move_to(
            [RIGHT_EDGE, S.KICKER_Y, 0], aligned_edge=RIGHT
        )
        cap.say(
            "The search plays the game forward using only Allie: no engine, no database.",
            *fade(k, kl, b, root, count, unit, key, head, names),
            shown.animate.set_value(1.0),
        )
        first = VGroup(named("Rd7+", "Rd7+"), named("Bxb7", "Bxb7"))
        self.add(first)
        self.play(
            *cap("Each leaf is one read of the network: what gets played next, and how the game ends."), grow(8, 2.0)
        )
        cap.hold()
        cap.say("The first 8 leaves read the 8 likeliest moves. Bxb7 looks fine.")
        cost = lambda line: P["tree"]["cost"][tree.node(line)]  # noqa: E731
        moves = [arrow(b, f, t, S.HILITE, 5) for f, t in (("c6", "b7"), ("c8", "c2"), ("f2", "e3"), ("e7", "d6"))]
        budget = lambda: 2 ** e.get_value()  # noqa: E731
        p2 = always_redraw(lambda: tree.path("Bxb7 Rc2+", budget()))
        rc2 = named("Bxb7 Rc2+", "Rc2+")
        self.add(p2, rc2)
        self.play(
            *cap(f"Leaf {cost('Bxb7 Rc2+')} reads Black's likeliest reply: Rc2+. Bxb7's score drops."),
            grow(16, 1.4),
            *fade(moves[0], moves[1]),
        )
        cap.hold()
        p3 = always_redraw(lambda: tree.path("Bxb7 Rc2+ Ke3 Kxd6", budget()))
        ke3, kxd6 = named("Bxb7 Rc2+ Ke3", "Ke3"), named("Bxb7 Rc2+ Ke3 Kxd6", "Kxd6")
        self.remove(p2)
        self.add(p3, ke3, kxd6)
        self.play(
            *cap(
                f"Leaf {cost('Bxb7 Rc2+ Ke3')} reads Ke3, leaf {cost('Bxb7 Rc2+ Ke3 Kxd6')} Kxd6. By 64 leaves, the tree has found the refutation."
            ),
            grow(64, 2.4),
            *fade(moves[2], moves[3]),
        )
        cap.hold(0.5)
        gone = VGroup(*moves, p3, first, rc2, ke3, kxd6)
        gone.clear_updaters()
        share = [MOVE[m]["leaves"]["1024"] for m in ("Rd7+", "Bxb7")]
        ends = VGroup(
            at(f"{share[0]} under Rd7+", RIGHT_EDGE, 0.95, S.LABEL, S.INK, RIGHT),
            at(f"{share[1]} under Bxb7", RIGHT_EDGE, -1.75, S.LABEL, S.INK, RIGHT),
            at(f"{1024 - sum(share)} under {len(P['moves']) - 2} others", -1.75, 0.75, S.LABEL, S.INK),
        )
        self.play(
            *cap(
                "Below the root it grows where play is likely to go, each side favoring moves that look good. At the root it stays broad."
            ),
            FadeOut(gone, run_time=0.4),
            grow(1024, 6.0),
            FadeIn(ends, run_time=6.0),
        )
        for m in (tree, count, cur):
            m.clear_updaters()
        cap.hold()
        cap.say(
            f"Every position is read as two {VIEW}-rated players in a classical game with no clocks,\nso values barely depend on who is playing.",
            extra=0.4,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class Tilt(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("The tilt")
        formula = S.text(
            f'π(move) ∝ p(move) · exp(<span foreground="{S.HILITE}">β</span> · v(move))',
            S.HEAD,
            S.INK,
            font=S.MONO,
            markup=True,
        ).move_to([0.8, 2.78, 0])
        b = demo_board(3.1, (-4.95, 0.62))
        x0, scale = -1.95, 6.7
        rows = (("Rd7+", 1.15), ("Bxb7", -0.2))

        def rowset(rating):
            r = BY_RATING[rating]
            bars, nums = VGroup(), VGroup()
            for m, y in rows:
                for key, dy, color, ink in (("p", 0.22, S.DENSE, S.INK2), ("pi", -0.22, S.BRASS, S.INK)):
                    bars.add(hbar(x0, y + dy, scale * r[key][m], 0.38, color))
                    nums.add(at(pct(r[key][m]), x0 + scale + 0.14, y + dy, S.LABEL, ink))
            return bars, nums

        names = VGroup()
        for m, y in rows:
            names.add(at(m, x0 - 0.16, y + 0.22, S.BODY, S.INK, RIGHT, font=S.MONO))
            names.add(at(f"v {signed(BY_RATING[2000]['v'][m])}", x0 - 0.16, y - 0.22, S.LABEL, S.INK2, RIGHT))
        key = VGroup(chip(S.DENSE, "p: Allie's prediction"), chip(S.BRASS, "π: what the bot samples from")).arrange(
            RIGHT, buff=0.4
        )
        key.move_to([x0, 2.05, 0], aligned_edge=LEFT)
        rail = Plot((-1.95, 5.75, -1.72, -1.72), (800, 2600), (0, 1))
        ticks = VGroup(Line(rail.p(800, 0), rail.p(2600, 0), stroke_color=S.RULE, stroke_width=4))
        for r in BINS:
            ticks.add(
                Line(rail.p(r, 0) + UP * 0.08, rail.p(r, 0) + DOWN * 0.08, stroke_color=S.RULE, stroke_width=3),
                at(str(r), rail.x(r), -2.02, S.SMALL, S.MUTED, ORIGIN),
            )
        title = at("Playing as", x0, -1.28)
        mark = lambda r: rect(0.24, 0.24, S.HILITE).move_to(rail.p(r, 0))  # noqa: E731
        beta = lambda r: at(
            f"β = {BY_RATING[r]['beta']:.2f}", RIGHT_EDGE, -1.15, S.HEAD, S.HILITE, RIGHT, weight="SEMIBOLD"
        )  # noqa: E731
        (cur, nums), pin, bt = rowset(800), mark(800), beta(800)
        cap.say(
            "The rule multiplies the prediction by exp(βv),\nv being the searched score on a log-odds scale, then samples.",
            *fade(k, formula, b, names, key, cur, nums),
        )
        a, z = BY_RATING[800], BY_RATING[2600]
        cap.say(
            "Same position, nearly the same values at every rating. What changes is β.",
            *fade(ticks, title, pin, bt),
        )
        self.play(
            *cap(
                f"At 800, β is {a['beta']:.2f}: Bxb7 stays near {pct(a['pi']['Bxb7'])}. At 2600, β is {z['beta']:.2f}: Bxb7 falls to {pct(z['pi']['Bxb7'])}, while the prediction alone says {pct(z['p']['Bxb7'])}."
            ),
            run_time=S.T_MOVE,
        )
        self.wait(1.2)
        for r in BINS[1:]:
            bars, shown = rowset(r)
            self.play(Transform(cur, bars), Transform(pin, mark(r)), run_time=0.45)
            self.remove(nums, bt)
            nums, bt = shown, beta(r)
            self.add(nums, bt)
        cap.hold(0.8)
        law = S.text(
            f'<span foreground="{S.HILITE}">β</span> = β<sub>0</sub> · exp(γx) · (t / 10 s)<sup>δ+ηx</sup> + floor',
            S.HEAD,
            S.INK,
            font=S.MONO,
            markup=True,
        ).move_to([0, 2.1, 0])
        where = at(
            "x = (rating − 1700) / 1000     t = how long humans take here, from Allie's own think-time head",
            0,
            1.42,
            S.LABEL,
            S.INK2,
            ORIGIN,
        )

        def tile(f, b_, box):
            v = cell(f, b_)["beta"]
            j = int(np.searchsorted((0.1, 0.25, 0.5, 1, 2), v))
            box.set_fill(S.PANEL if j == 0 else S.SEQ[j - 1])
            return VGroup(box, S.text(f"{v:.2f}", S.LABEL, S.GROUND if j >= 4 else S.INK).move_to(box))

        labels, tiles = cells_grid(1.04, 0.56, -4.4, 0.42, tile)
        note = at(
            f"Median β per cell. β₀, γ, δ, η, floor: one set per time control, {Z['tests']['parameters']} constants fit on July 2026 games",
            -4.4 - 1.04 / 2,
            -2.28,
            S.SMALL,
            S.MUTED,
        )
        self.play(FadeOut(VGroup(b, names, key, cur, nums, ticks, title, pin, bt)), run_time=S.T_QUICK)
        self.play(
            formula.animate.scale(0.6).move_to([RIGHT_EDGE - 2.6, 3.5, 0]),
            *fade(law, where),
            *cap(
                "β follows one formula: higher for stronger players and where humans think longer, never below a floor."
            ),
            run_time=S.T_MOVE,
        )
        self.play(FadeIn(labels), LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.012), FadeIn(note), run_time=1.4)
        cap.hold()
        cap.say("The blind spot's shape again. In bullet, the search barely counts.", extra=0.3)
        self.play(*wipe(self, cap, keep=[k]), run_time=S.T_MOVE)
        frame, origin = road_frame()
        path, end = sharpened()
        pl = ROAD_PLOT
        road = pl.line(
            *zip(
                *[(ROAD["raw"]["blunder"], 0.0)] + [(t["blunder"], t["ce"]) for t in ROAD["tilt"] if t["scale"] <= 1.0]
            ),
            S.BRASS,
            6,
        )
        s = ROAD["shipped"]
        dot = rect(0.24, 0.24, S.BRASS).move_to(pl.p(s["blunder"], s["ce"]))
        word = S.text(
            f"The rule: {s['blunder']:.1f}%, {signed(s['ce'], 3)} nats", S.LABEL, S.INK, weight="SEMIBOLD"
        ).next_to(dot, RIGHT, buff=0.15)
        self.play(
            *fade(frame, origin, path, end),
            *cap(
                "Back to classical 2400: tilting reaches the humans' strength, and human moves get less surprising, not more."
            ),
            run_time=S.T_MOVE,
        )
        self.play(Create(road), run_time=1.4)
        self.play(FadeIn(dot), FadeIn(word), run_time=S.T_QUICK)
        cap.hold(0.3)
        T, i = G["typical"], BINS.index(2400)
        cap.say(
            f"That position was picked to show the mechanism.\nThe median classical 2400 position shifts {pct(T['classical'][i], 1)} of its probability; bullet 2400, {pct(T['bullet'][i], 1)}.",
            extra=0.4,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class Live(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("One live move")
        L, Q = Z["live"], P["think_quantiles"]
        draw, low, cost, margin, searched = (
            Q["0.5"],
            Q["0.1"],
            Z["rule"]["cost"],
            Z["rule"]["margin"],
            P["search_seconds"],
        )
        steps = VGroup(
            *[
                S.text(s, S.BODY, S.INK, weight="SEMIBOLD", line_spacing=0.9).move_to(
                    [LEFT_EDGE, y, 0], aligned_edge=LEFT
                )
                for s, y in (
                    ("Draw a\nthink time", 1.85),
                    ("Pick a\nsearch size", 0.05),
                    ("Search, tilt,\nsample, wait", -1.72),
                )
            ]
        )
        # the think-time head per log second, so the wider bins from 16 s on don't show as a bump
        probs = np.array(P["time_probs"])
        edges = np.r_[np.arange(1, 16) - 0.5, 15.5, 16 * np.exp((np.arange(16, 63) - 16 + 0.5) / 7.06)]
        widths = np.diff(np.log(edges))
        widths[15] = 1 / 7.06  # bin 16 starts below 15.5 s in the draw (behaviour.think), overlapping bin 15
        dens = probs[1:] / widths
        pl = Plot((-3.3, RIGHT_EDGE, 0.98, 2.55), (0.5, 300), (0, dens.max() * 1.02), xlog=True)
        hist = VGroup(
            *[
                rect(pl.x(edges[i + 1]) - pl.x(edges[i]) - 0.012, max(pl.y(dens[i]) - pl.y(0), 0.004), S.DENSE).move_to(
                    [(pl.x(edges[i]) + pl.x(edges[i + 1])) / 2, pl.y(0), 0], aligned_edge=DOWN
                )
                for i in range(len(dens))
                if edges[i + 1] <= 300
            ]
        )
        ticks = VGroup(
            *[
                at(s, pl.x(v), 0.78, S.SMALL, S.MUTED, ORIGIN)
                for v, s in (
                    (1, "1 s"),
                    (2, "2 s"),
                    (5, "5 s"),
                    (10, "10 s"),
                    (30, "30 s"),
                    (60, "1 min"),
                    (120, "2 min"),
                )
            ]
        )

        def pin_at(t, color=S.HILITE):
            return VGroup(
                Line(pl.p(t, 0), [pl.x(t), 3.0, 0], stroke_color=color, stroke_width=5),
                S.text(f"{t:.1f} s", S.BODY, color, weight="SEMIBOLD").next_to([pl.x(t), 2.82, 0], RIGHT, buff=0.1),
            )

        cw = (RIGHT_EDGE + 3.3 - 8 * 0.05) / 9
        xs = [-3.3 + cw / 2 + i * (cw + 0.05) for i in range(9)]
        ladder = VGroup()
        for x, n, key in zip(xs, ["none", *map(str, RUNGS[:-1]), "1,024"], ["0", *map(str, RUNGS)]):
            s = P["rung_share"][key]
            ladder.add(
                VGroup(
                    rect(cw, 0.56).move_to([x, 0.1, 0]),
                    at(n, x, 0.1, S.LABEL, S.INK, ORIGIN),
                    at(pct(s, 0 if s >= 0.1 else 1), x, -0.33, S.SMALL, S.MUTED, ORIGIN),
                )
            )
        ladder_note = at(
            "Under each size: how often this position's draws land there. A search must also fit a tenth of the clock.",
            -3.3,
            -0.66,
            S.SMALL,
            S.MUTED,
        )
        rung = lambda t: max((r for r in RUNGS if r * cost <= t - margin), default=0)  # noqa: E731
        pick, small = (
            outline(cw, 0.56, S.HILITE, 5).move_to(ladder[-1][0]),
            outline(cw, 0.56, S.MUTED, 3).move_to(ladder[RUNGS.index(rung(low)) + 1][0]),
        )
        tl = Plot((-3.3, RIGHT_EDGE, -1.7, -1.7), (0, draw), (0, 1))
        track = hbar(tl.box[0], -1.7, tl.box[1] - tl.box[0], 0.46, S.PANEL)
        fill = hbar(tl.box[0], -1.7, tl.x(searched) - tl.box[0], 0.46, S.BRASS)
        inside = S.text(f"{RUNGS[-1]:,} leaves in {searched:.1f} s", S.LABEL, S.GROUND).move_to(fill)
        foot = VGroup(
            at(
                f"A move without search: {L['ms_per_move']:.0f} ms. Batched in a search: {L['batched_leaf_ms'][0]} to {L['batched_leaf_ms'][1]} ms a leaf. {P['threads']} CPU threads",
                tl.box[0],
                -2.22,
            ),
            at(f"Move played at {draw:.1f} s", tl.box[1], -2.22, S.LABEL, S.INK, RIGHT),
        )
        off = S.tag(f"{minus(L['offset'])}: measured on Allie 2.0, not yet on Allie 2.1", warn=True).move_to(
            [RIGHT_EDGE, S.KICKER_Y, 0], aligned_edge=RIGHT
        )
        cap.say(
            f"The whole move, as the bot runs it. It plays as its opponent's rating minus {-L['offset']}.",
            *fade(k, off, steps[0], hist, ticks),
        )
        cap.say(
            f"Allie also predicts how long humans think here. The bot draws a time from that. Say {draw:.1f} s, the median draw.",
            FadeIn(pin_at(draw)),
        )
        cap.say(
            f"It takes the largest search that fits, budgeting {1000 * cost:g} ms a leaf: {draw:.1f} s buys {RUNGS[-1]:,} leaves, {low:.1f} s only {rung(low)}.",
            *fade(steps[1], ladder, ladder_note),
            Create(pick),
            FadeIn(pin_at(low, S.MUTED)),
            Create(small),
        )
        cap.say(
            "Then it searches, tilts, samples a move, and waits out the think time.",
            *fade(steps[2], track),
            GrowFromEdge(fill, LEFT),
            *fade(inside, foot),
            run_time=1.2,
            extra=0.2,
        )
        cap.say(
            f"Playing as {P['white']}, over all its draws, it plays Bxb7 {pct(P['played']['Bxb7'], 1)} of the time, against Allie alone's {pct(MOVE['Bxb7']['p'])}.",
            extra=0.3,
        )
        self.play(*wipe(self, cap, keep=[k]), run_time=S.T_MOVE)
        rows = ROAD["leaves"]
        pl2 = Plot((-5.5, 0.9, -1.85, 2.45), (6, 1300), (1, 5.2), xlog=True)
        fr = pl2.grid((1, 2, 3, 4, 5), (8, 32, 128, 512), yfmt=lambda v: f"{v}%", px=S.LABEL)
        fr.add(
            at("Blunder rate at the rule's β, classical 2400", LEFT_EDGE, 2.88),
            at("leaves", -2.3, -2.5, S.SMALL, S.MUTED, ORIGIN),
        )
        curve = pl2.line([r["leaves"] for r in rows], [r["blunder"] for r in rows], S.BRASS, 6)
        rawy, humy = ROAD["raw"]["blunder"], ROAD["human"]["blunder"]
        refs = VGroup(
            DashedLine(pl2.p(6, rawy), pl2.p(1300, rawy), stroke_color=S.DENSE, stroke_width=3),
            Line(pl2.p(6, humy), pl2.p(1300, humy), stroke_color=S.INK, stroke_width=4),
            S.text(f"Allie alone {rawy:.1f}%", S.LABEL).next_to(pl2.p(1300, rawy), DOWN + LEFT, buff=0.08),
            S.text(f"Humans {humy:.1f}%", S.LABEL, S.INK).next_to(pl2.p(1300, humy), UP + LEFT, buff=0.08),
        )
        table = VGroup(
            at("median leaves", 3.75, 2.45, S.SMALL, S.MUTED),
            at("median search", RIGHT_EDGE, 2.45, S.SMALL, S.MUTED, RIGHT),
        )
        for i, f in enumerate(FMTS):
            y = 1.82 - 0.86 * i
            med = [cell(f, b)["leaves"] for b in BINS]
            secs = L["by_format"].get(f)
            table.add(
                at(f.capitalize(), 1.75, y, S.BODY, S.INK, weight="SEMIBOLD"),
                at(f"{min(med):,}" if min(med) == max(med) else f"{min(med):,}-{max(med):,}", 3.75, y, S.BODY, S.INK),
            )
            table.add(
                at(
                    "no test" if secs is None else f"{secs:.1f} s",
                    RIGHT_EDGE,
                    y,
                    S.BODY,
                    S.MUTED if secs is None else S.HILITE,
                    RIGHT,
                )
            )
        table.add(
            S.text(
                f"Four test games at a human pace.\nAll moves: median {L['search_median']:.1f} s, 90th percentile {L['search_p90']:.1f} s",
                S.SMALL,
                S.MUTED,
                line_spacing=0.9,
            ).move_to([1.75, -1.72, 0], aligned_edge=LEFT)
        )
        cap.say(
            f"Bigger trees catch more blunders, most of them in the first {rows[4]['leaves']} leaves.",
            *fade(fr, refs),
            Create(curve),
            run_time=1.6,
        )
        cap.say(
            f"Fast games get small trees. At a human pace, the median search took {L['search_median']:.1f} s.",
            FadeIn(table),
            extra=0.3,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class Close(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("How close")
        w, gap, x0 = 2.92, 0.26, -5.72
        acc = [Plot((x0 + i * (w + gap), x0 + i * (w + gap) + w, 0.68, 2.42), (700, 2700), (83, 97)) for i in range(4)]
        blu = [
            Plot((x0 + i * (w + gap), x0 + i * (w + gap) + w, -1.98, -0.25), (700, 2700), (0.8, 14), ylog=True)
            for i in range(4)
        ]
        frames = VGroup(at("Rating of the player to move", 0, -2.55, S.SMALL, S.MUTED, ORIGIN))
        for i, f in enumerate(FMTS):
            frames.add(acc[i].grid((85, 90, 95), yfmt=lambda v: f"{v}%", left=i == 0))
            frames.add(blu[i].grid((1, 3, 10), (1000, 1800, 2600), yfmt=lambda v: f"{v}%", left=i == 0))
            frames.add(
                at(f.capitalize(), (acc[i].box[0] + acc[i].box[1]) / 2, 2.75, S.BODY, S.INK2, ORIGIN, weight="SEMIBOLD")
            )
        sides = VGroup(
            S.text("Move accuracy", S.SMALL, S.INK2).rotate(np.pi / 2).move_to([-6.85, 1.55, 0]),
            S.text("Blunder rate", S.SMALL, S.INK2).rotate(np.pi / 2).move_to([-6.85, -1.12, 0]),
        )

        def series(key, i, j):
            return [cell(FMTS[i], b)[key][j] for b in BINS]

        humans = VGroup()
        for i in range(4):
            for j, pl in ((0, acc[i]), (1, blu[i])):
                y, se = np.array(series("human", i, j)), np.array(series("human_se", i, j))
                humans.add(
                    pl.band(BINS, np.maximum(y - 1.96 * se, pl.yr[0]), y + 1.96 * se, S.INK, 0.16),
                    pl.line(BINS, y, S.INK, 3.5),
                )

        def marks(key, color, size, turn=0.0):
            return VGroup(
                *[
                    rect(size, size, color).rotate(turn).move_to(pl.p(b, v))
                    for i in range(4)
                    for j, pl in ((0, acc[i]), (1, blu[i]))
                    for b, v in zip(BINS, series(key, i, j))
                ]
            )

        raw, ship = marks("raw", S.DENSE, 0.1, np.pi / 4), marks("calibrated", S.BRASS, 0.12)
        legend = VGroup(
            VGroup(
                Line(LEFT * 0.2, RIGHT * 0.2, stroke_color=S.INK, stroke_width=4), S.text("Humans", S.LABEL, S.INK)
            ).arrange(RIGHT, buff=0.1),
            VGroup(rect(0.13, 0.13, S.DENSE).rotate(np.pi / 4), S.text("Allie alone", S.LABEL, S.INK)).arrange(
                RIGHT, buff=0.1
            ),
            VGroup(rect(0.13, 0.13, S.BRASS), S.text("Calibrated, as the bot plays", S.LABEL, S.INK)).arrange(
                RIGHT, buff=0.1
            ),
        ).arrange(RIGHT, buff=0.4)
        legend.move_to([RIGHT_EDGE, S.KICKER_Y, 0], aligned_edge=RIGHT)
        cap.say(
            "How close does it get? September 2026 games, none used in the fit, though earlier candidates were scored on them. The line is the humans.",
            *fade(k, frames, sides, humans, legend[0]),
        )
        cap.say(
            "Allie alone falls below them as rating and thinking time grow.",
            LaggedStart(*[FadeIn(m) for m in raw], lag_ratio=0.004),
            FadeIn(legend[1]),
            run_time=1.2,
        )
        cap.say(
            "Calibrated play follows them from 800 to 2600, in every time control.",
            LaggedStart(*[FadeIn(m) for m in ship], lag_ratio=0.004),
            FadeIn(legend[2]),
            run_time=1.2,
        )
        rings = VGroup()
        for m in Z["misses"]:
            f, b = m["cell"].split("/")
            i, j = FMTS.index(f), 0 if m["test"] == "acc" else 1
            rings.add(
                outline(0.26, 0.26, S.FLAG, 3).move_to(
                    (acc if j == 0 else blu)[i].p(int(b), cell(f, int(b))["calibrated"][j])
                )
            )
        T = Z["tests"]
        cap.say(
            f"But not everywhere: {T['sept']} of the {T['of']} tests fall outside the noise, where about {T['chance']} would by chance.",
            LaggedStart(*[Create(r) for r in rings], lag_ratio=0.08),
            run_time=1.2,
            extra=0.4,
        )
        self.play(*wipe(self, cap, keep=[k]), run_time=S.T_MOVE)
        R = Z["rms"]
        table = VGroup(at("Typical miss against the humans, root-mean-square Elo", LEFT_EDGE, 2.5, S.BODY, S.INK))
        for i, f in enumerate(FMTS):
            y = 1.62 - 1.0 * i
            table.add(at(f.capitalize(), LEFT_EDGE, y, S.BODY, S.INK))
            for dy, v, color, ink in ((0.2, R["raw"][f], S.DENSE, S.INK2), (-0.2, R["calibrated"][f], S.BRASS, S.INK)):
                table.add(hbar(-4.2, y + dy, 0.03 * v, 0.36, color), at(str(v), -4.06 + 0.03 * v, y + dy, S.BODY, ink))
        key = (
            VGroup(chip(S.DENSE, "Allie alone"), chip(S.BRASS, "Calibrated"))
            .arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            .move_to([RIGHT_EDGE, 2.2, 0], aligned_edge=RIGHT)
        )
        rr = list(R["raw"].values())
        cap.say(
            f"The typical miss falls from {min(rr)}-{max(rr)} Elo to {R['calibrated']['classical']} in classical and {R['calibrated']['bullet']} in bullet.",
            *fade(table, key),
        )
        cap.say(
            f"Bullet's flat human curve inflates Elo, but that hides a regression: on the same curve and games, Allie 2.0 misses by {R['allie20']['bullet']} there.",
            extra=0.4,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class Short(Scene):
    def construct(self):
        S.ground(self)
        cap = S.Captions(self)
        k = S.kicker("Where it falls short")
        T, C = Z["tests"], Z["ce"]
        x = (LEFT_EDGE, -3.3, -1.6, 0.05)
        left = VGroup(
            at(f"Strength tests missed, of {T['of']}", x[0], 2.72, S.BODY, S.INK),
            S.text("July,\nheld out", S.SMALL, S.MUTED, line_spacing=0.9).move_to([x[1], 2.08, 0], aligned_edge=LEFT),
            at("September", x[2], 2.0, S.SMALL, S.MUTED),
            at("Constants", x[3], 2.0, S.SMALL, S.MUTED),
        )
        for y, name, a, b, c in (
            (1.45, "Allie 2.1", T["july"], T["sept"], T["parameters"]),
            (0.8, "Allie 2.0", T["allie20"]["july"], T["allie20"]["sept"], T["allie20"]["parameters"]),
        ):
            left.add(
                at(name, x[0], y, S.BODY),
                at(str(a), x[1], y, S.HEAD, S.INK),
                at(str(b), x[2], y, S.HEAD, S.INK),
                at(str(c), x[3], y, S.HEAD, S.MUTED),
            )
        lo, hi = T["candidates_sept"]
        left.add(
            at(
                f"About {T['chance']} would miss by chance. September also scored 3 earlier candidates: {lo}-{hi}",
                x[0],
                0.22,
                S.SMALL,
                S.MUTED,
            ).scale(0.92, about_edge=LEFT)
        )
        xr = (1.2, 3.45, 4.5, 5.55, 6.6)

        def by_format(title, y, rows, show):
            """A heading, the time controls, then a row per model, all over the right half."""
            out = VGroup(at(title, xr[0], y, S.BODY, S.INK))
            for xx, f in zip(xr[1:], FMTS):
                out.add(at(f.capitalize(), xx, y - 0.46, S.SMALL, S.MUTED, RIGHT))
            for k, (name, vals) in enumerate(rows):
                yy = y - 0.87 - 0.4 * k
                out.add(at(name, xr[0], yy, S.BODY))
                for xx, f in zip(xr[1:], FMTS):
                    out.add(at(show(vals[f]), xx, yy, S.LABEL, S.INK, RIGHT, font=S.MONO))
            return out

        right = by_format(
            "Surprise against Allie alone, September, nats",
            2.72,
            (("Allie 2.1", C["allie21"]), ("Allie 2.0", C["allie20"])),
            lambda v: signed(v, 4),
        )
        R = Z["rms"]
        miss = by_format(
            "Typical miss against the humans, RMS Elo",
            0.96,
            (("Allie 2.1", R["calibrated"]), ("Allie 2.0", R["allie20"])),
            str,
        )
        xb = (LEFT_EDGE, -3.9, -1.9, 0.1, 2.1, 4.1)
        low = VGroup(at("Worst cells, too weak: move accuracy %, blunder rate %", xb[0], -0.42, S.BODY, S.INK))
        for xx, h in zip(xb[1:], ("Humans", "Allie alone", "Calibrated", "Elo gaps", "At β 0.44")):
            low.add(at(h, xx, -0.95, S.SMALL, S.MUTED))
        for y, f in ((-1.42, "bullet"), (-2.02, "blitz")):
            c = cell(f, 800)
            low.add(at(f"{f.capitalize()} 800", xb[0], y, S.BODY))
            for xx, key in zip(xb[1:4], ("human", "raw", "calibrated")):
                low.add(at(f"{c[key][0]:.1f}, {c[key][1]:.1f}", xx, y, S.LABEL, S.INK))
            low.add(at(f"{minus(c['elo'][0])}, {minus(c['elo'][1])}", xb[4], y, S.LABEL, S.HILITE))
            hit = next(r for r in G["tilt_reach"][f"{f}/800"] if r["beta"] == 0.44)
            low.add(at(f"{hit['accuracy']:.1f}, {hit['blunder']:.1f}", xb[5], y, S.LABEL, S.INK))
        b8 = cell("bullet", 800)
        a, z = T["one_rule_july"]
        cap.say("Allie 2.1 misses more tests than Allie 2.0 did, even with more constants.", FadeIn(k), FadeIn(left))
        cap.say(
            f"Not like for like: its {T['july']} is the best of {T['variants']} variants on those July games. With a single rule, it misses {a} to {z}."
        )
        cap.say(
            "The gain in prediction is small, and smaller than Allie 2.0's. On July, bullet 2000 predicts worse than Allie alone, game halves pooled.",
            FadeIn(right),
            extra=0.2,
        )
        cap.say(
            f"Its typical miss is wider too, on the same curve and games: {R['calibrated']['bullet']} Elo against {R['allie20']['bullet']} in bullet, {R['calibrated']['blitz']} against {R['allie20']['blitz']} in blitz, {R['calibrated']['rapid']} against {R['allie20']['rapid']} in rapid.",
            FadeIn(miss),
        )
        cap.say(
            f"At bullet 800, Allie alone is {b8['human'][0] - b8['raw'][0]:.1f} accuracy points short. The rule's one bullet formula leaves β at {b8['beta']:.2f} here, too low to close it.",
            FadeIn(low),
        )
        cap.say(f"The search stops at {RUNGS[-1]:,} leaves: larger ones cost too much GPU time to fit.")
        cap.say(
            f"And the live {minus(Z['live']['offset'])} is Allie 2.0's. Allie 2.1's own hasn't been measured.",
            extra=0.3,
        )
        self.play(*wipe(self, cap), run_time=S.T_MOVE)


class End(Scene):
    def construct(self):
        S.ground(self)
        g = VGroup(
            S.title("Play it", S.TITLE),
            S.text("lichess.org/@/AllieTheChessBot", S.HEAD, S.HILITE, font=S.MONO),
            S.text("github.com/y0mingzhang/allie", S.BODY, S.INK2, font=S.MONO),
            S.text("Stockfish scored these moves. No engine evaluation ever trained Allie,\nthough a small share of its training games are engine games.", S.BODY, S.INK2, line_spacing=0.9),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0, 0.1, 0])  # fmt: skip
        self.play(FadeIn(g), run_time=S.T_MOVE)
        self.wait(4.0)
        self.play(FadeOut(g), run_time=S.T_MOVE)
