"""How Allie Works: one real game, read by the released Allie 2.1. Every number comes from data.json, which says where
each came from. render.sh renders the scenes in VIDEO['scenes'] and joins them."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Rectangle,
    Scene,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import parts as P  # noqa: E402
import style as S  # noqa: E402

D = json.loads((HERE / "data.json").read_text())
G, PLIES, FOCUS, FACTS = D["game"], D["plies"], D["focus"], D["facts"]
K = FOCUS["k"]

VIDEO = dict(
    scenes=["Question", "Tokens", "Inputs", "Attention", "Router", "Specialists", "Outputs", "Replay", "Score"],
    gif=(("Replay", 3.0), 9.0),
    poster=("Replay", 8.0),
)

HEADER_N = 11
X0 = -S.FRAME_W / 2 + S.MARGIN  # the left margin, shared with kickers and captions
WIDE = S.FRAME_W - 2 * S.MARGIN
RIGHT_X = 0.55  # the right column beside a board


def at_left(m, x, y):
    return m.move_to([x, y, 0], LEFT)


def label(s, px=S.LABEL, color=S.INK2, **kw):
    return S.text(s, px, color, **kw)


def big(s, color=S.INK, px=S.DISPLAY):
    return S.text(s, px, color, weight="SEMIBOLD")


def stat(number, words, color=S.INK, px=S.DISPLAY, words_px=S.LABEL):
    words = label(words, words_px, line_spacing=0.8)
    return VGroup(big(number, color, px), words).arrange(DOWN, aligned_edge=LEFT, buff=0.04)


def mover_name(k):
    return "White" if k % 2 == 0 else "Black"


def mover_elo(k):
    return G["white"] if k % 2 == 0 else G["black"]


def pct(p):
    return f"{100 * p:.0f}%" if p >= 0.01 else "<1%"


def lesion(name):
    return D["lesion"]["cases"][name]


def signed(x, digits=3):
    return f"{x:+.{digits}f}"


def tag_lines(*lines):
    """A tag of several lines."""
    t = S.text("\n".join(lines), S.SMALL, S.INK2, line_spacing=0.8)
    box = Rectangle(width=t.width + 0.34, height=t.height + 0.26, stroke_color=S.RULE, stroke_width=3)
    return VGroup(box, t.move_to(box))


def lesion_tag():
    return S.tag(
        f"Allie 2.1, changed after training: {D['lesion']['plies']:,} positions, a natural mix of September games"
    )


class Base(Scene):
    def setup(self):
        S.ground(self)
        self.cap = S.Captions(self, wps=2.8)

    def say(self, s, *anims, run_time=S.T_MOVE, extra=0.0):
        self.cap.say(s, *anims, run_time=run_time, extra=extra)

    def caption(self, s, *anims, run_time=S.T_MOVE):
        """The caption and animations, without waiting for it to be read."""
        self.play(*self.cap(s), *anims, run_time=run_time)

    def out(self, *mobs):
        self.play(*[FadeOut(m) for m in mobs], run_time=S.T_QUICK)

    def finish(self):
        self.cap.cur = None
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=S.T_QUICK)


def interpolate_grid(grid, v):
    """A {uci: p} between the two measured grid points around v; only the animation's in-betweens are interpolated."""
    keys = sorted(int(r) for r in grid)
    v = max(keys[0], min(keys[-1], v))
    hi = next(k for k in keys if k >= v)
    lo = keys[max(keys.index(hi) - 1, 0)]
    t = 0.0 if hi == lo else (v - lo) / (hi - lo)
    a, c = grid[str(lo)], grid[str(hi)]
    return {u: (1 - t) * a.get(u, 0) + t * c.get(u, 0) for u in set(a) | set(c)}


class Slider(VGroup):
    """A value on a track with its name, ticks, a knob and a readout; times on a log scale, ratings linear."""

    def __init__(self, x, y, w, grid, ticks, start, name, fmt, log=False):
        super().__init__()
        self.lo, self.hi, self.x, self.w = grid[0], grid[-1], x, w
        self.f, self.g = (math.log, math.exp) if log else (float, float)
        self.v = ValueTracker(self.f(start))
        self.add(Line([x, y, 0], [x + w, y, 0], stroke_color=S.RULE, stroke_width=4))
        self.add(VGroup(*[label(fmt(t), S.SMALL).move_to([self.at(t), y - 0.37, 0]) for t in ticks]))
        self.add(at_left(label(name), x, y + 0.37))
        self.knob = always_redraw(lambda: P.box(0.14, 0.36, S.HILITE).move_to([self.at(self.value()), y, 0]))
        self.read = always_redraw(
            lambda: S.text(fmt(int(round(self.value()))), S.BODY, S.INK, font=S.MONO).move_to(
                [x + w, y + 0.37, 0], RIGHT
            )
        )
        self.add(self.knob, self.read)

    def at(self, v):
        return self.x + self.w * (self.f(v) - self.f(self.lo)) / (self.f(self.hi) - self.f(self.lo))

    def value(self):
        return self.g(self.v.get_value())

    def to(self, v):
        return self.v.animate.set_value(self.f(v))

    def freeze(self):
        self.knob.clear_updaters()
        self.read.clear_updaters()


def dist_rows(dist, n=5):
    """The n likeliest moves of a {uci: p} as (san, p), always with the move really played."""
    top = sorted(dist, key=lambda u: -dist[u])
    played = G["uci"][K]
    ids = [played] + [u for u in top if u != played]
    return [(FOCUS["san"][u], dist.get(u, 0.0)) for u in sorted(ids[:n], key=lambda u: -dist.get(u, 0.0))]


def live_view(b, grid, value, n, x, y):
    """Bars and arrows that follow a sweep over a measured grid, top of the bars at (x, y)."""

    def view(dist):
        bars = P.move_bars(dist_rows(dist, n), 3.7)
        return bars.move_to(bars.get_center() - bars.get_corner(UP + LEFT) + np.array([x, y, 0])), P.arrows(b, dist)

    bars = always_redraw(lambda: view(interpolate_grid(grid, value()))[0])
    arrows = always_redraw(lambda: view(interpolate_grid(grid, value()))[1])
    return bars, arrows


class Question(Base):
    def construct(self):
        cap = self.cap
        b = S.board(G["fens"][K], 5.3).move_to([-3.75, 0.28, 0])
        real = FOCUS["real"]
        head = at_left(S.title("What will this player do?"), RIGHT_X, 2.7)
        who = S.text(
            f"{mover_name(K)} to move, rated {mover_elo(K)}, {P.clock_text(FOCUS['left'])} left", S.BODY, S.INK2
        )
        who.next_to(head, DOWN, buff=0.22, aligned_edge=LEFT)
        self.add(S.kicker("how allie 2.1 works"))
        scope = S.tag("Every number is the released Allie 2.1's, unless marked").next_to(
            who, DOWN, buff=0.3, aligned_edge=LEFT
        )
        self.play(FadeIn(b), run_time=S.T_MOVE)
        self.say(
            "A real blitz game from September 2026, the month after Allie's training data ends.",
            FadeIn(head),
            FadeIn(who),
            FadeIn(scope),
        )
        self.say("An engine looks for the best move. Allie asks what this person will play.")

        slider = Slider(
            RIGHT_X + 0.15,
            -2.0,
            5.2,
            sorted(int(r) for r in FOCUS["rating"]),
            range(800, 2801, 400),
            mover_elo(K),
            f"{mover_name(K)}'s rating",
            str,
        )
        sweep = dict(FOCUS["rating"]) | {str(mover_elo(K)): real}  # the real rating's own numbers
        bars, arrows = live_view(b, sweep, slider.value, 5, RIGHT_X, 1.45)
        picked = at_left(
            tag_lines("Picked because rating moves it a lot;", "a typical position moves about a quarter as much"),
            RIGHT_X,
            -1.1,
        )
        self.say("Its answer: a probability for every legal move.", FadeOut(scope), FadeIn(bars), FadeIn(arrows))
        self.say("Change one thing: this player's rating. Same board, same game.", FadeIn(slider), FadeIn(picked))
        self.play(slider.to(800), run_time=1.6)
        self.caption("Rated 800 to 1400: it expects the plain recapture, fxe6.")
        cap.hold()
        self.play(slider.to(2800), run_time=3.2, rate_func=linear)
        self.caption("Rated 2200 and up: Qxh4, over 90%.")
        cap.hold()
        self.play(slider.to(mover_elo(K)), run_time=1.0)
        bars.clear_updaters()
        arrows.clear_updaters()
        slider.freeze()

        played, p = G["uci"][K], PLIES[K]["p"]
        keep = P.arrow(b.at(played[:2]), b.at(played[2:4]), 0.13, S.HILITE, 1.0)
        self.remove(bars)
        marked = P.move_bars(dist_rows(real, 5), 3.7, highlight=G["san"][K])
        marked.move_to(marked.get_center() - marked.get_corner(UP + LEFT) + np.array([RIGHT_X, 1.45, 0]))
        self.add(marked)
        self.say(
            f"The real player, with {FOCUS['left']} seconds left, played {G['san'][K]} in {G['thinks'][K]} seconds. Allie gave it {pct(p)}.",
            FadeOut(arrows),
            FadeOut(picked),
            FadeIn(keep),
        )
        formula = VGroup(
            label("surprise"), S.text(f"-ln {p:.3f} = {-math.log(p):.2f} nats", S.HEAD, S.INK, font=S.MONO)
        )
        at_left(formula.arrange(DOWN, aligned_edge=LEFT, buff=0.1), RIGHT_X, -1.75)
        self.say(
            "So it was surprised. We measure surprise as minus the log of that probability.",
            FadeOut(slider),
            FadeIn(formula),
        )
        self.say("Averaged over many moves, surprise is its loss. Lower is better.")
        self.finish()


class Tokens(Base):
    def construct(self):
        head = ["start", str(G["tc"][0]), str(G["tc"][1]), *f"{G['white']:04d}", *f"{G['black']:04d}"]
        kinds = ["header"] * 3 + ["rating"] * 8 + ["white" if i % 2 == 0 else "black" for i in range(K)]
        page = P.chips(head + G["uci"][:K], kinds, WIDE).move_to([X0, S.KICKER_Y - 0.5, 0], UP + LEFT)
        hdr = page[:HEADER_N]
        target = hdr.copy()
        hdr.scale(2.0).move_to([0, 0.75, 0])
        labels = VGroup()
        for lo, hi, s in ((1, 2, "time control, seconds"), (3, 6, "White's rating"), (7, 10, "Black's rating")):
            g = hdr[lo : hi + 1]
            rule = Line(
                g.get_corner(DOWN + LEFT) + 0.1 * DOWN,
                g.get_corner(DOWN + RIGHT) + 0.1 * DOWN,
                stroke_color=S.RULE,
                stroke_width=3,
            )
            labels.add(VGroup(rule, label(s, S.SMALL).next_to(rule, DOWN, buff=0.1)))
        self.add(S.kicker("a game is a line of tokens"))
        self.say(
            "Allie reads a game as a line of tokens: first two for the time control, then both ratings, digit by digit.",
            LaggedStart(*[FadeIn(c) for c in hdr], lag_ratio=0.12),
            FadeIn(labels),
            run_time=2.0,
        )
        self.play(FadeOut(labels), Transform(hdr, target), run_time=S.T_MOVE)
        b = S.board(size=3.35).move_to([5.0, -0.95, 0])
        note = at_left(stat("1,968", "possible move tokens", px=S.TITLE), X0, -0.9)
        self.say("Then one token per move: from-square and to-square.", FadeIn(b), FadeIn(note))
        for i in range(K):
            self.play(FadeIn(page[HEADER_N + i]), *P.play_move(b, G["uci"][i]), run_time=0.18 if i < 10 else 0.08)
        slot = P.chip("?", "rating", width=page[-1][0].width).next_to(page[-1], RIGHT, buff=0.07)
        self.say(
            "The next token is the move to predict, as a language model predicts the next word.",
            FadeIn(slot),
            FadeOut(note),
        )
        stats = VGroup(
            *[
                stat(a, c, px=S.HEAD)
                for a, c in (
                    (FACTS["tokens_read"], "tokens read in training"),
                    (FACTS["lichess_games"], "Lichess games to draw from"),
                    (FACTS["otb_games"], "over-the-board games"),
                )
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        at_left(stats, X0, -1.05)
        self.say(
            "It learned this way on real games of every time control. Engine games are under 1% of its tokens.",
            FadeIn(stats),
        )
        self.finish()


PIECE_ORDER = "PNBRQKpnbrqk"


def planes_of(fen):
    """Square indices (a1 = 0) of each piece kind in a FEN's placement, plus the empty squares last."""
    out = {c: [] for c in PIECE_ORDER}
    for i, row in enumerate(fen.split()[0].split("/")):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                out[c].append((7 - i) * 8 + f)
                f += 1
    full = {s for v in out.values() for s in v}
    return [out[c] for c in PIECE_ORDER] + [[s for s in range(64) if s not in full]]


def hand_angle(seconds, k):
    """The model's clock feature k of a time as an angle: its sin and cos of pi 2^k ln(1 + t) / 10 (Model.clock)."""
    return math.pi * 2**k * math.log1p(max(seconds, 0)) / 10


class Inputs(Base):
    def construct(self):
        cap, parts = self.cap, D["parts"]
        common = parts["scale"]
        code = lambda name, w, h=0.42, scale=common: P.barcode(parts[name], w, h, scale)
        self.add(S.kicker("what allie sees at each move"))

        # the token
        y = 0.45
        c = P.chip(G["uci"][K - 1], "white" if (K - 1) % 2 == 0 else "black", S.HEAD, h=0.62).move_to([-5.45, y, 0])
        c_l = label("the last move").next_to(c, UP, buff=0.14)
        table = VGroup(
            Rectangle(width=1.1, height=2.3, stroke_color=S.RULE, stroke_width=3),
            *[
                Line([-0.55, v, 0], [0.55, v, 0], stroke_color=S.RULE, stroke_width=1)
                for v in np.linspace(-1.15, 1.15, 25)[1:-1]
            ],
        ).move_to([-3.2, y - 2.3 / 48, 0])
        row = P.box(1.1, 2.3 / 24, S.HILITE).move_to([-3.2, y, 0])
        t_l = label("2,432 rows").next_to(table, UP, buff=0.14)
        vec = code("embed", 7.2, scale=parts["embed_scale"]).move_to([2.95, y, 0])
        v_l = label("2,048 learned numbers, every 4th drawn").next_to(vec, UP, buff=0.14)
        link = Line(
            c.get_right() + 0.08 * RIGHT, [table.get_left()[0] - 0.08, y, 0], stroke_color=S.RULE, stroke_width=4
        )
        link2 = Line(
            row.get_right() + 0.05 * RIGHT, vec.get_left() + 0.05 * LEFT, stroke_color=S.HILITE, stroke_width=4
        )
        first = VGroup(c, c_l, link, table, t_l, row, link2, vec, v_l)
        self.say(
            "Each token looks up a learned vector of 2,048 numbers. This is the real one for White's last move.",
            FadeIn(first),
        )
        self.out(first)

        # the board
        size, pitch = 1.16, 1.36
        planes = VGroup()
        for i, squares in enumerate(planes_of(G["fens"][K])):
            col, line = (i, 0) if i < 6 else (i - 6, 1) if i < 12 else (6, 0.5)
            pl = P.plane(squares, size).move_to([X0 + size / 2 + col * pitch, 1.75 - line * 1.72, 0])
            mark = S.piece(PIECE_ORDER[i], 0.42) if i < 12 else label("empty", S.SMALL)
            planes.add(VGroup(pl, mark.next_to(pl, UP, buff=0.06)))
        cnn = at_left(
            label("13 planes, then 3 convolution layers of 32 channels: the board's 2,048 numbers"), X0, -1.22
        )
        vec = code("board", WIDE).move_to([0, -1.72, 0])
        extra = VGroup(label("and whose move it is"), label("castling rights"), label("en passant")).arrange(
            DOWN, aligned_edge=LEFT, buff=0.1
        )
        extra.move_to([3.6, 1.6, 0], LEFT)
        self.say(
            "A token says little alone, so Allie also gets the whole board, read by a small convolutional network.",
            LaggedStart(*[FadeIn(p) for p in planes], lag_ratio=0.06),
            FadeIn(cnn),
            FadeIn(vec),
            FadeIn(extra),
            run_time=1.4,
        )
        self.out(planes, cnn, vec, extra)

        # the clocks: three times, each on eight clock hands
        ck = FOCUS["clocks"]
        dial_row = lambda seconds, y: VGroup(
            *[P.dial(0.42, hand_angle(seconds, k)).move_to([-2.0 + k * 1.02, y, 0]) for k in range(8)]
        )
        reading = lambda seconds, y: S.text(P.clock_text(seconds), S.HEAD, S.INK, font=S.MONO).move_to(
            [-2.7, y, 0], RIGHT
        )
        rows = VGroup()
        for i, (name, seconds) in enumerate(
            (("time left", ck["left"]), ("opponent's", ck["other"]), ("last think", ck["prev"]))
        ):
            y = 2.1 - i * 1.18
            rows.add(VGroup(at_left(label(name, S.BODY), X0, y), reading(seconds, y), dial_row(seconds, y)))
        formula = at_left(
            S.text("hand k: angle = pi 2^k ln(1 + seconds) / 10", S.LABEL, S.INK2, font=S.MONO), -2.42, -1.6
        )
        self.say(
            "Then the clocks: its time, its opponent's, its last think. Each time is read off eight clock hands.",
            FadeIn(rows),
            FadeIn(formula),
        )
        t = ValueTracker(ck["left"])
        live = VGroup(
            always_redraw(lambda: reading(t.get_value(), 2.1)), always_redraw(lambda: dial_row(t.get_value(), 2.1))
        )
        self.remove(rows[0][1], rows[0][2])
        self.add(live)
        self.caption(
            "Slow hands give the rough size, fast hands fine detail, on a log scale: the last seconds count most.",
            t.animate.set_value(180),
            run_time=1.2,
        )
        self.play(t.animate.set_value(1), run_time=3.6, rate_func=linear)
        cap.hold()
        self.play(t.animate.set_value(ck["left"]), run_time=0.8)
        for m in live:
            m.clear_updaters()
        self.out(rows[0][0], rows[1], rows[2], live, formula)

        # what the clock does to this position
        b = S.board(G["fens"][K], 4.5).move_to([-4.3, 0.5, 0])
        grid = {r: v for r, v in FOCUS["clock"].items() if int(r) <= 180}
        slider = Slider(
            RIGHT_X,
            -1.75,
            5.2,
            sorted(int(r) for r in grid),
            (2, 10, 60, 180),
            180,
            "time left",
            P.clock_text,
            log=True,
        )
        bars, arrows = live_view(b, grid, slider.value, 3, RIGHT_X, 2.3)
        self.add(b, arrows, bars)
        self.play(FadeIn(slider), FadeIn(b), FadeIn(bars), FadeIn(arrows), run_time=S.T_QUICK)
        self.say(f"Back to our position. With 3 minutes left, Qxh4 is {pct(grid['180']['d8h4'])}.")
        self.play(slider.to(10), run_time=2.2, rate_func=linear)
        self.caption(
            f"With 10 seconds, the reflex recapture goes from {pct(grid['180']['f7e6'])} to {pct(grid['10']['f7e6'])}."
        )
        cap.hold()
        self.play(slider.to(2), run_time=1.0, rate_func=linear)
        self.say(f"At 2 seconds it falls back to {pct(grid['2']['f7e6'])}.")
        Rc = D["recapture"]
        like = at_left(
            tag_lines(
                f"{Rc['positions']} contested blitz recaptures: up {100 * Rc['rise_10s']:.1f} points [{100 * Rc['rise_ci'][0]:.1f}, {100 * Rc['rise_ci'][1]:.1f}]",
                f"from 3 minutes to 10 s, and {100 * Rc['drop_2s']:.1f} more at 2 s",
            ),
            RIGHT_X,
            -0.1,
        )
        self.say(
            "Nobody wrote this in. Across similar positions the rise averages about 4 points; the drop at 2 seconds is this position's own.",
            FadeIn(like),
        )
        bars.clear_updaters()
        arrows.clear_updaters()
        slider.freeze()
        self.out(b, bars, arrows, slider, like)

        no = lesion("no_clock")
        low = no["slices"]["under 30 s"]
        st = VGroup(
            stat(signed(no["ce_diff"]), "loss, clocks hidden", px=S.TITLE),
            stat(signed(low["diff"]), "when under 30 s are left", px=S.TITLE),
            stat(
                signed(FACTS["loss_21"] - FACTS["loss_20"], 4), "for scale: Allie 2.0 to 2.1, main test set", px=S.TITLE
            ),
        ).arrange(RIGHT, buff=0.8, aligned_edge=UP)
        at_left(st, X0, 1.5)
        notes = VGroup(
            lesion_tag(),
            S.tag("10% of its training games had no clocks, so it knows this state"),
            S.tag(f"same test on a small model, main test set, a different mix: {signed(FACTS['clock_small'])}"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        at_left(notes, X0, -0.75)
        self.say(
            "Hide the clocks from the trained model and the loss rises, more when time is short.",
            FadeIn(st),
            FadeIn(notes),
        )
        self.out(st, notes)

        # they add up
        stack = VGroup()
        for i, (name, key) in enumerate(
            (("move", "embed"), ("board", "board"), ("clocks", "clock"), ("this token", "sum"))
        ):
            y = 2.05 - i * 1.1
            lab = S.text(name, S.BODY, S.INK if i == 3 else S.INK2).move_to([-3.75, y, 0], RIGHT)
            op = (
                S.text("" if i == 0 else "=" if i == 3 else "+", S.BODY, S.INK2).move_to([-3.45, y, 0])
                if i
                else VGroup()
            )
            stack.add(VGroup(lab, op, code(key, 9.7, 0.56).move_to([1.85, y, 0])))
        self.say(
            "The three are added into one vector, and that enters the first block.",
            LaggedStart(*[FadeIn(s) for s in stack], lag_ratio=0.3),
            FadeIn(label("one colour scale; every 4th number shown", S.SMALL).move_to([6.7, -1.75, 0], RIGHT)),
            run_time=1.4,
        )
        self.finish()


def page_tokens(n_moves):
    head = ["start", str(G["tc"][0]), str(G["tc"][1]), *f"{G['white']:04d}", *f"{G['black']:04d}"]
    return head + G["uci"][:n_moves]


def paint(page, weights, scale=1.0):
    """Animations colouring chips by attention weight."""
    out = []
    for c, w in zip(page, weights):
        a = w * scale
        out += [
            c[0].animate.set_fill(S.mix(S.PANEL, S.HILITE, min(1.0, a) ** 0.55), 1),
            c[1].animate.set_color(S.BLACK_SIDE if a > 0.3 else S.INK),
        ]
    return out


class Attention(Base):
    def construct(self):
        cap, A = self.cap, D["attention"]
        self.add(S.kicker("attention: looking back"))
        names = page_tokens(K + 2)
        page = P.chips(names, ["header"] * len(names), WIDE, px=30, h=0.52, line=0.66).move_to(
            [X0, S.KICKER_Y - 0.5, 0], UP + LEFT
        )
        for c in page:
            c[1].set_color(S.INK2)
        shown = VGroup(*page[: HEADER_N + K])
        query = lambda i: Rectangle(
            width=page[i].width + 0.08, height=page[i].height + 0.08, stroke_color=S.INK, stroke_width=4
        ).move_to(page[i])
        q = query(HEADER_N + K - 1)
        self.say(
            "Now the token can look back. Attention reads any earlier token of this game, and nothing later.",
            FadeIn(shown),
            Create(q),
        )
        y = -0.55
        share = VGroup(
            big(f"{100 * A['rating_share']:.0f}%", S.HILITE),
            label(
                f"of all attention goes to the 8 rating digits, {100 * A['header_share']:.0f}% to the whole header",
                S.BODY,
            ),
        ).arrange(RIGHT, buff=0.4)
        at_left(share, X0, y - 0.2)
        even = at_left(
            S.tag(
                f"Spread evenly over the tokens it can see, they would get {100 * A['even_rating']:.0f}% and {100 * A['even_header']:.0f}%. Over {A['games']} September games"
            ),
            X0,
            y - 1.1,
        )
        self.say(
            "Ratings appear only once, at the very start. At move 20, attention still reads them directly.",
            *paint(shown, A["mean_at_focus"], 5),
            FadeIn(share),
            FadeIn(even),
            run_time=S.T_SLOW,
            extra=1.0,
        )
        self.out(share, even)

        def card(ex):
            g = VGroup(
                S.text(f"block {ex['block']}, head {ex['head']}", S.BODY, S.INK, font=S.MONO),
                label(f"reads {ex['label']}", S.BODY),
            )
            return at_left(g.arrange(DOWN, aligned_edge=LEFT, buff=0.14), X0, y - 0.2)

        ex = A["examples"]
        c0 = card(ex[0])
        xsa = at_left(
            tag_lines(
                "Exclusive self-attention: each head also rescales its projection on the token's own value",
                f"by a learned amount, usually reducing it; {100 * A['xsa_negative']:.0f}% of heads add to it",
            ),
            X0,
            -1.95,
        )
        self.say(
            "Each block has 32 attention heads.",
            *paint(shown, [0] * len(shown)),
            FadeIn(xsa),
        )
        self.say(
            f"This one puts {100 * A['head47_mover']:.0f}% of its attention on the rating of whoever is to move.",
            *paint(shown, ex[0]["weights"][str(K)]),
            FadeIn(c0),
            run_time=S.T_SLOW,
        )
        self.caption("One move later, it has jumped to the other player's rating. And back.", run_time=S.T_QUICK)
        for j in (1, 2):
            i = HEADER_N + K - 1 + j
            shown.add(page[i])
            self.play(
                FadeIn(page[i]), Transform(q, query(i)), *paint(shown, ex[0]["weights"][str(K + j)]), run_time=S.T_MOVE
            )
            self.wait(0.7)
        cap.hold()
        c = c0
        for e, s in zip(
            ex[1:],
            (
                f"Another reads the time control: {100 * A['time_control_head']:.0f}% of its attention.",
                f"Another, the player's own last move: {100 * A['own_last_head']:.0f}%.",
            ),
        ):
            nxt = card(e)
            self.say(s, FadeOut(c), FadeIn(nxt), *paint(shown, e["weights"][str(K + 2)]))
            c = nxt
        few = at_left(
            S.tag(
                f"{A['head_top_over_half']} of 768 heads put at least half their attention on one of six token groups; for the median head, the top group gets {100 * A['head_top_median']:.0f}%"
            ),
            X0,
            -1.95,
        )
        self.say(
            "A few heads are this specific. These are among the sharpest of 768.", FadeOut(xsa), FadeIn(few), extra=0.4
        )
        self.play(FadeOut(c), FadeOut(few), *paint(shown, [0] * len(shown)), run_time=S.T_QUICK)

        no, cut = lesion("no_ratings"), lesion("last_8_plies")
        far = round(D["lesion"]["ratings_by_mover"]["bands"][-1]["distance"], -2)
        table = VGroup(
            label("loss, all players", S.SMALL).move_to([1.2, 0.42, 0], LEFT),
            VGroup(
                label(f"players rated 2400+, {no['slices']['2400+']['n']:,} positions,", S.SMALL),
                label(f"about {far} points from the median", S.SMALL),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            .move_to([3.1, 0.52, 0], LEFT),
        )
        for i, (name, c) in enumerate(
            (("Both ratings set to the median", no), ("Attention limited to the header and last 8 plies", cut))
        ):
            yy, top = -0.02 - i * 0.56, c["slices"]["2400+"]
            table.add(
                at_left(S.text(name, S.BODY, S.INK), X0, yy),
                S.text(signed(c["ce_diff"]), S.BODY, S.INK, font=S.MONO).move_to([1.2, yy, 0], LEFT),
                S.text(
                    f"{signed(top['diff'])} [{top['ci'][0]:.3f}, {top['ci'][1]:.3f}]", S.BODY, S.INK, font=S.MONO
                ).move_to([3.1, yy, 0], LEFT),
            )
        notes = VGroup(
            lesion_tag(),
            S.tag(
                f"same rating test on a small model, main test set, different mix and median: {signed(FACTS['rating_small'])}, {signed(FACTS['rating_small_2400'])}"
            ),
            S.tag("the limit is on direct attention: older moves still flow through the tokens in between"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        at_left(notes, X0, -1.84)
        self.say(
            "Set both ratings to the median and the loss rises, the more the further a player is from it.",
            FadeIn(table),
            FadeIn(notes),
        )
        self.say("Limit attention to the header and the last 8 plies, and it rises too.", extra=0.6)
        self.finish()


def score_level(s):
    return next((i for i, t in enumerate((0.9, 0.7, 0.5, 0.3, 0.1)) if s >= t), None)


def all_blocks(size, pitch, top_left):
    """The 24 blocks in 4 rows of 6: three plain squares, then an expert grid for each of the 21 routed blocks."""
    cell = lambda i: np.array(
        [top_left[0] + (i % 6) * pitch + size / 2, top_left[1] - (i // 6) * pitch - size / 2, 0.0]
    )
    plain = VGroup(*[VGroup(P.box(size, size, S.RULE), label("plain", S.SMALL)).move_to(cell(i)) for i in range(3)])
    grids = [P.Grid(size, color=S.mix(S.PANEL, S.RULE, 0.7)).move_to(cell(i)) for i in range(3, 24)]
    return VGroup(plain, *grids), grids, cell


class Router(Base):
    def construct(self):
        cap, R = self.cap, D["route"]
        f, row = R["focus"], R["focus"]["row"]
        self.add(S.kicker("the router"))
        blocks, grids, cell = all_blocks(1.06, 1.3, (-0.85, 3.0))
        words = VGroup(
            big("24", px=S.TITLE),
            label("blocks, one after another", S.BODY),
            label("each: attention, then a network", S.BODY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        at_left(words, X0, 1.7)
        at = ValueTracker(0)
        ring = always_redraw(
            lambda: Rectangle(width=1.2, height=1.2, stroke_color=S.HILITE, stroke_width=5).move_to(
                cell(min(int(at.get_value()), 23))
            )
        )
        self.say(
            "The token passes through 24 transformer blocks: attention, then a small network, in each.",
            FadeIn(words),
            FadeIn(blocks),
            run_time=S.T_QUICK,
        )
        self.add(ring)
        self.play(at.animate.set_value(24), run_time=1.6, rate_func=linear)
        ring.clear_updaters()
        self.remove(ring)
        self.say(
            "The first three networks are plain. The other 21 hold 256 small experts each, drawn here as a 16 by 16 grid."
        )

        grid = P.Grid(4.9).move_to([1.15, 0.2, 0])
        title = S.text(f"block {f['block']}: 256 experts", S.BODY, S.INK).next_to(
            grid, UP, buff=0.14, aligned_edge=LEFT
        )
        self.play(FadeOut(blocks), FadeOut(words), FadeIn(grid), FadeIn(title), run_time=S.T_MOVE)

        # scores, then the 16
        vec = P.barcode(D["parts"]["router_input"], 2.6, 0.32, D["parts"]["router_scale"]).move_to([-5.3, 0.2, 0])
        vec_l = label(f"this token, by block {f['block']}").next_to(vec, UP, buff=0.1)
        link = Line(vec.get_right() + 0.08 * RIGHT, grid.get_left() + 0.08 * LEFT, stroke_color=S.RULE, stroke_width=4)
        route_l = label("router").next_to(link, UP, buff=0.08)
        levels = [[e for e, s in enumerate(f["score"]) if score_level(s) == lv] for lv in range(5)]
        ramp = list(reversed(S.SEQ))
        heat = VGroup(*[grid.cells(ids, ramp[lv]) for lv, ids in enumerate(levels) if ids])
        into = VGroup(vec, vec_l, link, route_l)
        self.say(
            "A router scores all 256 for this token: one learned vector per expert, a dot product, a sigmoid.",
            FadeIn(into),
            FadeIn(heat),
        )
        lit = grid.cells(f["ids"], S.HILITE)
        starved = at_left(
            tag_lines(
                "Capped biases weren't enough in Allie 2.0's first",
                "attempt: 116 of 256 experts starved. Router centring",
                "plus Adam on every step fixed it",
            ),
            X0,
            -1.25,
        )
        self.say(
            "Only the top 16 run, with a per-expert bias to spread the load. The other 240 do nothing for this token.",
            FadeOut(heat),
            FadeIn(lit),
            FadeIn(starved),
            extra=1.0,
        )
        shared = P.box(4.9, 0.34, S.HILITE).next_to(grid, DOWN, buff=0.14)
        shared_l = S.text("shared expert", S.SMALL, S.BLACK_SIDE).move_to(shared)
        self.say("A shared expert runs for every token.", FadeOut(starved), FadeIn(shared), FadeIn(shared_l))

        # gates
        gates, yy, total = VGroup(), 2.65, sum(f["gates"])
        for gate in sorted(f["gates"], reverse=True):
            h = 4.9 * gate / total
            gates.add(P.box(0.9, max(h - 0.03, 0.02), S.HILITE).move_to([4.3, yy - h / 2, 0]))
            yy -= h
        g_l = (
            VGroup(label("16 weights"), label("sum to 4"))
            .arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            .next_to(gates, RIGHT, buff=0.16)
        )
        self.say(
            "Their outputs are added, weighted by their scores. The weights always sum to 4.",
            FadeIn(gates),
            FadeIn(g_l),
        )

        # the next moves
        self.out(gates, g_l, into)
        read = at_left(S.text(f"after {G['san'][K - 1]}", S.BODY, S.INK2, font=S.MONO), X0, 1.5)
        self.add(read)
        lo, hi = D["scan"]["overlap_next_range"]
        self.caption(
            f"Next token, different experts: in this block it shares {D['scan']['overlap_next']:.1f} of the 16 on average ({lo:.1f} to {hi:.1f} across blocks), {D['scan']['overlap_same_side']:.1f} with the same player's next turn. Chance: 1.",
            run_time=S.T_QUICK,
        )
        for j in range(1, 8):
            self.remove(lit, read)
            lit = grid.cells(R["idx"][K + j][row], S.HILITE)
            read = at_left(S.text(f"after {G['san'][K + j - 1]}", S.BODY, S.INK2, font=S.MONO), X0, 1.5)
            self.add(lit, read)
            self.wait(0.5)
        cap.hold()
        self.out(grid, title, lit, shared, shared_l, read)

        # the arithmetic: same compute, 12 times the units
        F = FACTS
        unit = WIDE / (F["hidden_expert"] * 256 + F["hidden_shared"])
        bar = lambda x, n, y, c: P.box(unit * n, 0.46, c).move_to([x + unit * n / 2, y, 0])
        used, shared_n = 16 * F["hidden_expert"], F["hidden_shared"]
        a = VGroup(
            at_left(S.text("A plain block: 5,456 hidden units, all used", S.BODY, S.INK), X0, 2.45),
            bar(X0, F["hidden_dense"], 1.95, S.DENSE),
        )
        b = VGroup(
            at_left(S.text("A routed block, per token: 16 x 256 + 1,360 = 5,456", S.BODY, S.INK), X0, 1.05),
            bar(X0, used, 0.55, S.HILITE),
            bar(X0 + unit * used, shared_n, 0.55, S.BRASS),
        )
        c = VGroup(
            at_left(S.text("A routed block, stored: 256 x 256 + 1,360 = 66,896", S.BODY, S.INK), X0, -0.35),
            bar(X0, 256 * F["hidden_expert"], -0.85, S.PANEL),
            bar(X0, used, -0.85, S.HILITE),
            bar(X0 + unit * 256 * F["hidden_expert"], shared_n, -0.85, S.BRASS),
        )
        self.say(
            "A plain block's network has 5,456 hidden units. A routed block uses just as many per token, so it costs about the same.",
            FadeIn(a),
            FadeIn(b),
        )
        self.say("But it stores 12 times as many, and each token picks the ones it needs.", FadeIn(c))
        self.out(a, b, c)
        totals = VGroup(
            stat(F["total"], "parameters stored", words_px=S.BODY),
            stat(F["active"], "used for each move", S.HILITE, words_px=S.BODY),
        ).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        at_left(totals, X0, 2.0)
        share = float(F["active"][:-1]) / float(F["total"][:-1])
        whole = VGroup(
            P.box(WIDE, 0.46).move_to([0, 0.62, 0]),
            P.box(WIDE * share, 0.46, S.HILITE).move_to([X0 + WIDE * share / 2, 0.62, 0]),
        )
        dense = at_left(
            S.text(
                f"To match in training, a plain model needs {F['dense_vs_moe'][0]} to {F['dense_vs_moe'][1]} times the compute",
                S.BODY,
                S.INK,
            ),
            X0,
            -0.65,
        )
        sweep = at_left(
            S.tag(
                f"fitted, not measured: two laws, on budgets {F['dense_budget_gap'][0]} to {F['dense_budget_gap'][1]} times below Allie 2.1's; intervals {F['dense_interval'][0]} to {F['dense_interval'][1]}"
            ),
            X0,
            -1.3,
        )
        self.say(
            f"So the model stores {F['total']} parameters and uses {F['active']} per move.",
            FadeIn(totals),
            FadeIn(whole),
        )
        self.say(
            "On the smaller models we swept, fitted laws say a plain model needs over twice the training compute to match. Untested at this size.",
            FadeIn(dense),
            FadeIn(sweep),
        )
        self.out(totals, whole, dense, sweep)

        # do the choices matter?
        rows = [
            ("The router's 16", lesion("full")["ce"], S.ALLIE21),
            ("Its best 8", lesion("best_8")["ce"], S.DENSE),
            ("Its best 4", lesion("best_4")["ce"], S.DENSE),
            ("Its best 1", lesion("best_1")["ce"], S.DENSE),
            ("Shared expert only", lesion("shared_only")["ce"], S.DENSE),
            ("16 random experts", lesion("random_experts")["ce"], S.FLAG),
        ]
        top = max(v for _, v, _ in rows)
        table = VGroup()
        for i, (name, v, color) in enumerate(rows):
            yy = 2.6 - i * 0.72
            w = 7.8 * v / top
            table.add(
                at_left(S.text(name, S.BODY, S.INK), X0, yy),
                P.box(w, 0.42, color).move_to([-2.5 + w / 2, yy, 0]),
                at_left(
                    S.text(f"{v:.2f}" if color == S.FLAG else f"{v:.4f}", S.BODY, S.INK, font=S.MONO),
                    -2.5 + w + 0.16,
                    yy,
                ),
            )
        note = VGroup(
            lesion_tag(),
            VGroup(
                S.tag("Higher than the headline, which averages 16 cells equally"),
                S.tag("Best k: each keeps its original weight"),
            ).arrange(RIGHT, buff=0.14),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        at_left(note, X0, -1.98)
        self.say(
            "Do the choices matter? Give each token 16 random experts and the loss nearly doubles.",
            FadeIn(table),
            FadeIn(note),
        )
        self.say(
            f"Yet the top few do most of the work: its best 8 of the 16 cost only {lesion('best_8')['ce_diff']:.4f}, its best 4 only {lesion('best_4')['ce_diff']:.3f}.",
            extra=0.6,
        )
        self.finish()


class Specialists(Base):
    def construct(self):
        Sc, R = D["scan"], D["route"]
        n = len(PLIES)
        self.add(S.kicker("what the experts do"))

        # the whole game in one block, sorted by when experts usually run
        row, x0, y0, w, h = Sc["row"], -2.6, 3.0, 9.2, 5.3
        cw, ch = w / n, h / 256
        cells = lambda rank: [
            (x0 + k * cw, y0 - (rank[e] + 1) * ch, cw * 0.84, ch * 0.92) for k in range(n) for e in R["idx"][k][row]
        ]
        back = P.box(w, h).move_to([x0 + w / 2, y0 - h / 2, 0])
        raster = P.rects(cells(list(range(256))), S.HILITE)
        words = VGroup(
            S.text(f"block {Sc['block']}", S.BODY, S.INK, font=S.MONO),
            label("across: this game, move by move"),
            label("down: its 256 experts"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        at_left(words, X0, 2.45)
        self.say(
            f"Nobody told the experts what to do. Here is our whole game in block {Sc['block']}, 16 lit cells per move. It looks like noise.",
            FadeIn(back),
            FadeIn(raster),
            FadeIn(words),
        )
        self.say("Now sort the experts by when in a game they usually run, measured on other games.")
        self.play(Transform(raster, P.rects(cells(Sc["rank"]), S.HILITE)), run_time=2.2)
        self.say("The game sweeps through them. Some experts belong to the opening, others to the endgame.")
        self.out(back, raster, words)

        # two experts that track one thing, on the shown game
        for d, truth, first, k, square, s in zip(
            Sc["detectors"],
            (G["in_check"], G["pawn_g4"]),
            ("in check", "Black pawn on g4"),
            (G["in_check"][0], G["pawn_g4"][0]),
            ("g8", "g4"),
            (
                "Some have one clear job. This one runs exactly when the player to move is in check.",
                f"This one nearly does: it runs when a Black pawn is on g4. In our game, all {len(G['pawn_g4'])} positions from the move it lands, and no others.",
            ),
        ):
            b = P.tint(S.board(G["fens"][k], 4.4), [square], S.FLAG).move_to([-4.25, 0.6, 0])
            grid = P.Grid(2.45).move_to([0.25, 1.62, 0])
            j = D["route"]["blocks"].index(d["block"])
            lit = grid.cells(R["idx"][k][j], S.BRASS)
            one = grid.cells([d["expert"]], S.HILITE)
            ring = Rectangle(width=0.26, height=0.26, stroke_color=S.INK, stroke_width=4).move_to(
                grid.center_of(d["expert"])
            )
            name = S.text(f"block {d['block']}, expert {d['expert']}", S.LABEL, S.INK, font=S.MONO).next_to(
                grid, UP, buff=0.1, aligned_edge=LEFT
            )
            numbers = VGroup(
                stat(pct(d["rate"]), d["of"], S.HILITE, S.HEAD),
                stat(f"{100 * d['elsewhere']:.1g}%", "of all other positions", S.INK, S.HEAD),
                stat(pct(d["precision"]), d["runs"], S.INK, S.HEAD),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            at_left(numbers, 1.85, 1.42)
            tx, tw = -1.0, 7.6
            pitch = tw / n
            track = lambda ids, y, c: VGroup(
                P.rects([(tx + i * pitch, y - 0.15, pitch * 0.82, 0.3) for i in range(n)], S.PANEL),
                *([P.rects([(tx + i * pitch, y - 0.15, pitch * 0.82, 0.3) for i in ids], c)] if ids else []),
            )
            lit_k = [i for i in range(n) if d["expert"] in R["idx"][i][j]]
            rows = VGroup(
                at_left(label(first, S.SMALL), tx, -0.46),
                track(truth, -0.82, S.INK),
                at_left(label(f"expert {d['expert']} runs", S.SMALL), tx, -1.2),
                track(lit_k, -1.56, S.HILITE),
                at_left(label("move 1", S.SMALL), tx, -1.94),
                label(f"move {(n + 1) // 2}", S.SMALL).move_to([tx + tw, -1.94, 0], RIGHT),
            )
            frame = VGroup(b, grid, lit, one, ring, name, numbers, rows)
            self.say(s, FadeIn(frame), extra=0.6)
            self.out(frame)

        # others follow the player or the clock
        cards = VGroup()
        for c in Sc["cards"]:
            top = P.box(max(3.7 * c["rate"], 0.02), 0.32, S.HILITE)
            low = P.box(max(3.7 * c["base"], 0.02), 0.32, S.DENSE)
            top.move_to([top.width / 2, 0, 0])
            low.move_to([low.width / 2, -0.4, 0])
            nums = VGroup(
                S.text(pct(c["rate"]), S.LABEL, S.INK, font=S.MONO).next_to(top, RIGHT, buff=0.1),
                S.text(
                    f"{100 * c['base']:.1f}%" if c["base"] < 0.1 else pct(c["base"]), S.LABEL, S.INK2, font=S.MONO
                ).next_to(low, RIGHT, buff=0.1),
            )
            what = S.text(c["what"], S.LABEL, S.INK).move_to([0, 0.76, 0], LEFT)
            which = S.text(f"block {c['block']}, expert {c['expert']}", S.SMALL, S.INK2, font=S.MONO).move_to(
                [0, 0.4, 0], LEFT
            )
            runs = S.text(f"{pct(c['precision'])} of its runs on target", S.SMALL, S.INK2).move_to([0, -0.76, 0], LEFT)
            cards.add(VGroup(what, which, top, low, nums, runs))
        cards.arrange_in_grid(rows=2, cols=3, buff=(0.45, 0.3), cell_alignment=LEFT)
        at_left(cards, X0, 1.02)
        key = VGroup(
            P.box(0.24, 0.2, S.HILITE),
            label("runs in that situation", S.SMALL),
            P.box(0.24, 0.2, S.DENSE),
            label("runs overall", S.SMALL),
        ).arrange(RIGHT, buff=0.12)
        at_left(key, X0, -1.32)
        picked = at_left(
            S.tag(
                f"Each: the top F1 score on half of {Sc['positions']:,} September positions, measured on the other half"
            ),
            X0,
            -1.82,
        )
        self.say(
            "Others lean toward the player, the clock or the phase of the game.",
            FadeIn(cards),
            FadeIn(key),
            FadeIn(picked),
            extra=1.5,
        )
        close = S.tag(f"closely: F1 score above 0.8, over {Sc['features_tried']} features tried").next_to(
            picked, DOWN, buff=0.1, aligned_edge=LEFT
        )
        self.say(
            f"Most don't closely match any single feature we tried: only {Sc['f1_over_0.8']} of {Sc['experts']:,} do.",
            FadeIn(close),
            extra=0.4,
        )
        self.finish()


class Outputs(Base):
    def construct(self):
        ply, centres, think = PLIES[K], D["centres"], FOCUS["think"]
        self.add(S.kicker("what comes out"))
        b = S.board(G["fens"][K], 3.9).move_to([-4.55, 0.55, 0])
        who = label(f"{mover_name(K)} to move, rated {mover_elo(K)}").next_to(b, DOWN, buff=0.34)
        self.add(b, who)
        x0, W = -1.75, 8.3
        head = at_left(S.text("One last layer: 2,432 numbers", S.BODY, S.INK), x0, 2.95)
        self.say(
            "After block 24, one last layer turns the token's vector into 2,432 numbers. Three groups matter.",
            FadeIn(head),
        )

        # moves
        t1 = at_left(label("1,968 numbers: the next move", color=S.HILITE), x0, 2.42)
        rows = [(a, c) for a, _, c in ply["top"][:3]]
        bars = P.move_bars(rows, 5.6, label_w=1.2, row=0.4, px=S.LABEL)
        bars.move_to(bars.get_center() - bars.get_corner(UP + LEFT) + np.array([x0, 2.2, 0]))
        arr = P.arrows(b, {u: c for _, u, c in ply["top"]})
        self.say(
            "The next move: one number per move token. Only legal moves count when it plays.",
            FadeIn(t1),
            FadeIn(bars),
            FadeIn(arr),
        )

        # think time
        n, h, y = 34, 0.95, -0.72
        w = W / n
        t2 = at_left(
            label("63 numbers: how long it will think (training weight 0.2)", color=S.HILITE), x0, y + h + 0.58
        )
        tall = max(think[:n])
        hist = VGroup(
            *[
                P.box(w * 0.8, max(h * think[i] / tall, 0.012), S.BRASS).move_to(
                    [x0 + (i + 0.5) * w, y + h * think[i] / tall / 2, 0]
                )
                for i in range(n)
            ]
        )
        base = Line([x0, y, 0], [x0 + n * w, y, 0], stroke_color=S.RULE, stroke_width=3)
        x_of = lambda s: x0 + (min(range(n), key=lambda j: abs(centres[j] - s)) + 0.5) * w
        ticks = VGroup(*[label(f"{s} s", S.SMALL).move_to([x_of(s), y - 0.24, 0]) for s in (0, 5, 10, 15, 30, 60)])
        taken = G["thinks"][K]
        mark = Line([x_of(taken), y, 0], [x_of(taken), y + h + 0.06, 0], stroke_color=S.INK, stroke_width=4)
        mark_l = S.text(f"took {taken} s", S.SMALL, S.INK).next_to(mark, UP, buff=0.04)
        self.say(
            f"How long this player will think. It expected {ply['expect']:.0f} seconds on average. The player took {taken}.",
            FadeIn(t2),
            FadeIn(VGroup(hist, base, ticks)),
            FadeIn(mark),
            FadeIn(mark_l),
        )

        # result
        wn, dr, ls = ply["wdl"]
        y = -2.2
        t3 = at_left(
            label("3 numbers: how the game will end, for the mover (training weight 0.2)", color=S.HILITE), x0, y + 0.4
        )
        res, x = VGroup(), x0
        for v, c in zip((wn, dr, ls), (S.HILITE, S.DRAW_SIDE, S.BLACK_SIDE)):
            res.add(P.box(W * v, 0.36, c).move_to([x + W * v / 2, y, 0]))
            x += W * v
        vals = VGroup(
            S.text(f"win {pct(wn)}", S.LABEL, S.BLACK_SIDE).move_to([x0 + W * wn / 2, y, 0]),
            S.text(f"loss {pct(ls)}", S.LABEL, S.INK).move_to([x0 + W - W * ls / 2, y, 0]),
        )
        self.say(
            f"And how the game will end, for the player to move: win {pct(wn)}, draw {pct(dr)}, loss {pct(ls)}.",
            FadeIn(t3),
            FadeIn(res),
            FadeIn(vals),
        )
        self.say("Its training targets are what really happened: the move, the time, the result. No engine evaluation.")
        self.finish()


class Replay(Base):
    def construct(self):
        cap, R = self.cap, D["route"]
        n = len(PLIES)
        self.add(S.kicker("the whole game, as allie read it"))
        size = 4.5
        b = S.board(size=size, coords=False).move_to([-3.9, 0.42, 0])
        self.add(b)
        blocks, grids, _ = all_blocks(0.8, 0.96, (0.42, 3.2))
        self.add(blocks)
        bar_x = b.get_left()[0] - 0.24
        tag_y = (b.get_top()[1] + 0.2, b.get_bottom()[1] - 0.2)

        def clock(k, white):
            ks = [i for i in range(k) if (i % 2 == 0) == white]
            return P.clock_text(G["clocks"][ks[-1]] if ks else G["tc"][0])

        strip_x, strip_w, strip_y, tall = 0.42, 5.68, -2.46, 0.84
        unit = strip_w / n
        self.add(Line([strip_x, strip_y, 0], [strip_x + strip_w, strip_y, 0], stroke_color=S.RULE, stroke_width=2.5))
        self.add(at_left(label("surprise at each real move", S.SMALL), strip_x, -1.36))
        self.add(
            label("left bar: how it expects the game to end", S.SMALL).move_to(
                [b.get_right()[0], tag_y[1] - 0.36, 0], RIGHT
            )
        )
        dynamic = VGroup()
        self.add(dynamic)

        def frame(k):
            nonlocal dynamic
            self.remove(dynamic)
            q = PLIES[k]
            wn, dr, ls = q["wdl"]
            white, black = (wn, ls) if k % 2 == 0 else (ls, wn)
            dynamic = VGroup(
                *[g.cells(R["idx"][k][j], S.HILITE) for j, g in enumerate(grids)],
                P.eval_bar(white, dr, black, size).move_to([bar_x, b.get_center()[1], 0]),
                at_left(
                    S.text(f"Black {G['black']}  {clock(k, False)}", S.LABEL, S.INK, font=S.MONO),
                    b.get_left()[0],
                    tag_y[0],
                ),
                at_left(
                    S.text(f"White {G['white']}  {clock(k, True)}", S.LABEL, S.INK, font=S.MONO),
                    b.get_left()[0],
                    tag_y[1],
                ),
                S.text(f"{G['san'][k]}: {pct(q['p'])}", S.SMALL, S.INK, font=S.MONO).move_to(
                    [strip_x + strip_w, -1.36, 0], RIGHT
                ),
                P.arrows(b, {u: c for _, u, c in q["top"]}, floor=0.08),
            )
            self.add(dynamic)

        says = {
            0: "Each frame: its predictions, and the 336 experts it ran for that move, 16 in each routed block.",
            16: "The bars show its surprise at each real move, in nats.",
            30: f"Nxe5 surprised it: {pct(PLIES[30]['p'])}.",
            39: f"The move from the start. Qxh4 expected, fxe6 played: Black's chances fall from {pct(PLIES[39]['wdl'][0])} to {pct(PLIES[40]['wdl'][2])}.",
            52: f"Its top choice matched {D['top1_hits']} of the {n} moves. Mean surprise in this game: {D['mean_surprise']:.2f}.",
        }
        for k in range(n):
            frame(k)
            if k in says:
                self.caption(says[k], run_time=S.T_QUICK)
                if k in (30, 39):  # stay on the moves the captions are about
                    cap.hold()
            else:
                self.wait(0.2)
            h = tall * min(PLIES[k]["surprise"], 4.5) / 4.5
            self.add(
                P.box(unit * 0.74, max(h, 0.012), S.BRASS).move_to([strip_x + (k + 0.5) * unit, strip_y + h / 2, 0])
            )
            anims = P.play_move(b, G["uci"][k])
            self.bring_to_front(dynamic)
            self.play(*anims, run_time=0.2)
        cap.hold(extra=0.4)
        self.say(
            f"Black won. Allie's outlook swung with the game: White {pct(PLIES[44]['wdl'][0])} at move 23, {pct(PLIES[46]['wdl'][0])} at move 24."
        )
        self.finish()


def axis(x0, x1, y, lo, hi, ticks, fmt="{:.2f}"):
    """A horizontal value axis from data lo to hi over x0..x1. Returns the group and data -> x."""
    x_of = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    out = VGroup(Line([x0, y, 0], [x1, y, 0], stroke_color=S.RULE, stroke_width=3))
    for t in ticks:
        out.add(Line([x_of(t), y, 0], [x_of(t), y - 0.09, 0], stroke_color=S.RULE, stroke_width=3))
        out.add(label(fmt.format(t), S.SMALL).move_to([x_of(t), y - 0.3, 0]))
    return out, x_of


class Score(Base):
    def construct(self):
        F = FACTS
        self.add(S.kicker("how good it is"))
        ax, x_of = axis(-3.4, 6.2, 0.0, 1.2, 1.4, [1.2, 1.25, 1.3, 1.35, 1.4])
        rows = VGroup(ax)
        for i, (name, v, c) in enumerate(
            (
                ("Original Allie", F["loss_original"], S.ORIGINAL),
                ("Allie 2.0", F["loss_20"], S.ALLIE20),
                ("Allie 2.1", F["loss_21"], S.ALLIE21),
            )
        ):
            y = 2.45 - i * 0.82
            rows.add(
                Line([-3.4, y, 0], [6.2, y, 0], stroke_color=S.PANEL, stroke_width=3),
                S.text(name, S.BODY, S.INK).move_to([-3.75, y, 0], RIGHT),
                P.box(0.24, 0.24, c).move_to([x_of(v), y, 0]),
                S.text(f"{v:.4f}", S.BODY, S.INK, font=S.MONO).move_to([x_of(v), y + 0.3, 0]),
            )
        rows.add(
            label("trained on blitz only, scored here on every time control", S.SMALL).move_to(
                [x_of(F["loss_original"]) + 0.12, 2.17, 0], RIGHT
            )
        )
        note = at_left(
            label(
                "Loss on the main test set, lower is better: July 2026 games, held out of training, every time control and rating"
            ),
            X0,
            -0.72,
        )
        self.say(
            "On a month it never trained on, Allie 2.1's loss is 1.2240. Allie 2.0, the previous release: 1.2505.",
            FadeIn(rows),
            FadeIn(note),
        )
        tops = VGroup(
            stat(
                F["top1_blitz"],
                "on the blitz benchmark, four rating bands equally weighted:\nits top choice is the move played",
                S.HILITE,
            ),
            stat(F["top1_2400"], "in its band for players\nrated 2400 and above"),
        ).arrange(RIGHT, buff=0.8, aligned_edge=UP)
        at_left(tops, X0, -1.8)
        self.say("Humans are hard to predict: its top choice is right about six times in ten.", FadeIn(tops))
        self.out(rows, note, tops)

        # by rating: the same two models
        ax, x_of = axis(-2.7, 6.2, -1.72, 1.0, 1.5, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "{:.1f}")
        rows = VGroup(ax, S.text("loss", S.SMALL, S.INK2).move_to([-3.05, -2.02, 0], RIGHT))
        for i, (name, a, b) in enumerate(F["bands"]):
            y = 2.3 - i * 0.95
            rows.add(
                Line([-2.7, y, 0], [6.2, y, 0], stroke_color=S.PANEL, stroke_width=3),
                S.text(name, S.BODY, S.INK).move_to([-3.05, y, 0], RIGHT),
                Line([x_of(a), y, 0], [x_of(b), y, 0], stroke_color=S.INK2, stroke_width=4),
                P.box(0.22, 0.22, S.ALLIE20).move_to([x_of(a), y, 0]),
                P.box(0.22, 0.22, S.ALLIE21).move_to([x_of(b), y, 0]),
                S.text(f"{b:.3f}", S.LABEL, S.INK, font=S.MONO).move_to([x_of(b) - 0.6, y, 0]),
                S.text(signed(b - a), S.LABEL, S.INK2, font=S.MONO).move_to([x_of(a) + 0.65, y, 0]),
            )
        legend = (
            VGroup(
                P.box(0.2, 0.2, S.ALLIE21),
                S.text("Allie 2.1", S.LABEL, S.INK),
                P.box(0.2, 0.2, S.ALLIE20),
                S.text("Allie 2.0", S.LABEL, S.INK),
            )
            .arrange(RIGHT, buff=0.14)
            .move_to([3.6, 2.95, 0])
        )
        self.say(
            "Strong players are easier to predict than weak ones, and that is where Allie 2.1 gained most.",
            FadeIn(rows),
            FadeIn(legend),
        )
        self.out(rows, legend)

        # where it is weak
        head = at_left(S.title("Where it is weak", S.HEAD), X0, 2.85)
        lines = VGroup()
        for i, (a, b) in enumerate(F["weak"]):
            y = 2.2 - i * 0.82
            lines.add(VGroup(at_left(S.text(a, S.BODY, S.INK), X0, y + 0.13), at_left(label(b), X0, y - 0.19)))
        self.play(FadeIn(head), run_time=S.T_QUICK)
        self.caption("And where it falls short.", run_time=S.T_QUICK)
        for line in lines:
            self.play(FadeIn(line), run_time=S.T_QUICK)
            self.wait(2.6)
        self.out(head, lines)

        # cost
        both = VGroup(
            stat(F["ms"], "a move on a server CPU, 12 threads, no GPU", S.HILITE),
            stat(F["gb"], "of int8 weights read for that move"),
        ).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        at_left(both, X0, 0.9)
        train = at_left(label(F["trained"], S.BODY), X0, -0.9)
        self.say("Because each move reads so little of the model, it plays without a GPU.", FadeIn(both), FadeIn(train))
        self.out(both, train)

        # end card
        name = at_left(S.title("Allie 2.1", 120), X0, 1.6)
        links = VGroup(*[S.text(s, S.BODY, S.INK2, font=S.MONO) for s in F["links"]]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        )
        links.next_to(name, DOWN, buff=0.4, aligned_edge=LEFT)
        grid = P.Grid(4.2).move_to([4.2, 0.55, 0])
        lit = grid.cells(D["route"]["idx"][K][D["route"]["focus"]["row"]], S.HILITE)
        self.say(
            "Play it on Lichess, or download the weights. A companion video shows how it searches.",
            FadeIn(name),
            FadeIn(links),
            FadeIn(grid),
            FadeIn(lit),
        )
        self.wait(0.8)
