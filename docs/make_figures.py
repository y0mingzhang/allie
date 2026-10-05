"""Regenerate the README and DETAILS figures from the result files.

usage: uv run --extra figures python docs/make_figures.py [name ...]

Writes docs/figures/NAME.png (all figures by default) and prints the numbers each figure shows, so the text can
be checked against this output. Charts are Vega-Lite (Altair, rendered by vl-convert), the model diagram is
matplotlib; both use Inter from docs/fonts. The original Allie's scores are read from ALLIE_ORIGINAL (a directory
of its benchmark and rating-set scores); without them its marks are left out.
"""

import json
import math
import os
import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import vl_convert as vlc
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
X = R / "recipe10x"
DATA = Path(os.environ.get("ALLIE_DATA", "/data/group_data/dei-group/yimingz3/allie"))
ORIGINAL = Path(os.environ.get("ALLIE_ORIGINAL", DATA / "original-allie"))
OUT = ROOT / "docs" / "figures"
FONTS = ROOT / "docs" / "fonts"
FINAL = (
    R / "pretrain/bigfix-24x1536d75m4shipv2-s16-24x1536-c8s200f0v4-s42"
)  # the big run
ANN = "ann-all-p05-t1907"  # Allie 2.0: the big run's second anneal
SEARCH = DATA / "maia3-bench/search"
SWEEP = X / "sweep-readout-c8s200f0v4-w4.json"
C_ALLIE = (
    3.142088554411623e20  # the big run's useful training FLOPs (bigrun-trajectory-v2)
)
MAIA = {"maia3-5m": "Maia-3 5M", "maia3-23m": "Maia-3 23M", "maia3-79m": "Maia-3 79M"}

INK, INK2, MUTED = "#1f2328", "#57534e", "#78716c"
PAPER, GRID, NEUTRAL, NEUTRAL_DARK = "#f8f7f3", "#e7e4dc", "#c9c3b8", "#8a8378"
ALLIE, ALLIE_PALE, ALLIE_WASH = "#e4572e", "#f3b49f", "#f9d5c8"
MINUS = "−"
WIDTH = (
    634  # one-panel plot width; every figure renders about 730 px wide (1,460 at 2x)
)


@alt.theme.register("allie", enable=True)
def theme():
    return alt.theme.ThemeConfig(
        config={
            "font": "Inter",
            "background": PAPER,
            "padding": {"left": 20, "right": 24, "top": 20, "bottom": 16},
            "view": {"stroke": None},
            "title": {"anchor": "start", "fontSize": 18, "fontWeight": 700, "color": INK, "subtitleFontSize": 13,
                      "subtitleColor": INK2, "subtitlePadding": 6, "offset": 18, "subtitleLineHeight": 18},
            "axis": {"domain": False, "ticks": False, "gridColor": GRID, "labelColor": MUTED, "labelFontSize": 12.5,
                     "titleColor": INK2, "titleFontSize": 12, "titleFontWeight": 500, "titlePadding": 10,
                     "labelPadding": 6},
            "text": {"font": "Inter", "fontSize": 13.5, "color": INK2},
        }
    )  # fmt: skip


def jload(p):
    return json.loads(Path(p).read_text())


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def save(chart, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.png").write_bytes(vlc.vegalite_to_png(chart.to_json(), scale=2))
    print(f"wrote docs/figures/{name}.png")


def signed(v, nd):
    return f"{v:+.{nd}f}".replace("-", MINUS)


def power(v):
    """1e18 -> 10¹⁸."""
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "10" + str(round(math.log10(v))).translate(sup)


def power_axis():
    """Vega label expression writing powers of ten (16-22) as power() does."""
    expr = "''"
    for e in range(16, 23):
        expr = f"abs(log(datum.value) / log(10) - {e}) < 0.01 ? '{power(10.0**e)}' : ({expr})"
    return expr


def signed_axis(nd):
    return f"datum.value == 0 ? '0' : (datum.value > 0 ? '+' : '{MINUS}') + format(abs(datum.value), '.{nd}f')"


def title(text, *sub):
    return alt.TitleParams(text, subtitle=list(sub))


def panel_title(text):
    return alt.TitleParams(
        text, anchor="start", fontSize=13, fontWeight=600, color=INK, offset=10
    )


def values(rows):
    return alt.Chart(alt.Data(values=rows))


def diverging(rows, order, dom, nd, ticks, name, highlight, label_axis=True, width=300):
    """Horizontal bars of differences from a reference at zero, with whiskers (lo, hi) and signed labels."""
    b = values(rows)
    sx = alt.Scale(domain=dom, nice=False)
    y = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labels=label_axis, labelFontSize=13, labelFontWeight=500,
                                                               labelColor=INK, labelPadding=10))  # fmt: skip
    bar = b.mark_bar(height=16, cornerRadiusEnd=2).encode(
        y=y, color=alt.condition(highlight, alt.value(ALLIE), alt.value(NEUTRAL)),
        x=alt.X("d:Q", scale=sx, title=None, axis=alt.Axis(values=ticks, grid=True, labelExpr=signed_axis(nd))))  # fmt: skip
    whisk = b.mark_rule(color=INK, strokeWidth=1.3).encode(
        y=y, x=alt.X("lo:Q", scale=sx), x2="hi:Q"
    )
    pos = (
        b.transform_filter(alt.datum.d > 0)
        .mark_text(align="left", dx=7)
        .encode(y=y, x=alt.X("hi:Q", scale=sx), text="label:N")
    )
    neg = (
        b.transform_filter(alt.datum.d <= 0)
        .mark_text(align="right", dx=-7)
        .encode(y=y, x=alt.X("lo:Q", scale=sx), text="label:N")
    )
    zero = (
        values([{"v": 0}])
        .mark_rule(color=INK, strokeWidth=1.4)
        .encode(x=alt.X("v:Q", scale=sx))
    )
    return (bar + whisk + zero + pos + neg).properties(
        width=width, height=34 * len(order), title=panel_title(name)
    )


def allie_n():
    """Active and total non-embedding matmul parameters of Allie 2.0, and the big run's tokens."""
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT / "scripts"),
    ]  # this checkout's package, or its flat modules
    try:
        from allie.model import arch as model_arch
    except ImportError:
        import modded_arch as model_arch

    c = jload(FINAL / "resume-config.json")["config"]
    L, d = c["layers"], c["width"]
    e, k, routed, shared = model_arch.moe_dims(d, c["arch"])[:4]
    base = 4 * L * d * d + 3 * d * model_arch.swiglu_hidden(d) + (L - 1) * d * e
    act = base + (L - 1) * 3 * d * (shared + k * routed)
    tot = base + (L - 1) * 3 * d * (shared + e * routed)
    return act, tot, c["scheduled_steps"] * 524288


# ------------------------------------------------------------------------- the sweep's scaling laws


def macro():
    return next(m for m in jload(SWEEP)["metrics"] if m["metric"] == "macro")


def law(fam):
    """The sweep's L(N, D) = E + A (N/1e7)^-alpha + B (D/1e8)^-beta for the main eval."""
    t = macro()["law"][fam]["theta"]
    E, A, B = math.exp(t[0]), math.exp(t[1]), math.exp(t[3])
    return lambda n, d: E + A * (n / 1e7) ** -t[2] + B * (d / 1e8) ** -t[4]


def flops_per():
    """Training FLOPs per parameter and token, by size, as the sweep's runs measured it (experiments.readout);
    held at the end values outside the sweep's sizes."""
    ks = sorted((r["n"], r["c"] / (r["n"] * r["d"])) for r in jload(SWEEP)["runs"])
    return lambda n: np.interp(
        np.log(n), np.log([x for x, _ in ks]), [k for _, k in ks]
    )


def optimum(fam, c):
    """The law's compute-optimal (L*, N*, D*) at c training FLOPs, as experiments.readout.best finds it."""
    f, kap = law(fam), flops_per()
    o = minimize_scalar(
        lambda x: f(math.exp(x), c / (kap(math.exp(x)) * math.exp(x))),
        bounds=(math.log(1e5), math.log(c / 1e5)),
        method="bounded",
    )
    n = math.exp(o.x)
    return float(o.fun), n, c / (kap(n) * n)


def law_layers(x, y, key, lab):
    """Each family's law at its compute-optimal size (solid over the sweep, dashed up to the big run's compute),
    its isoflop minima, and a label per family at lab[fam] = (c, value)."""
    cells = macro()["cells"]
    layers = []
    for fam, color, name in (
        ("dense", NEUTRAL_DARK, "dense"),
        ("s16", ALLIE, "mixture of experts"),
    ):
        last = cells[fam]["1e18"]["c"]
        law_ = [
            dict(c=float(c), v=optimum(fam, c)[key])
            for c in np.geomspace(5e17, C_ALLIE, 60)
        ]
        mins = [
            dict(c=cells[fam][b]["c"], v=cells[fam][b]["l" if key == 0 else "n_opt"])
            for b in ("1e17", "3e17", "1e18")
        ]
        layers += [
            values([r for r in law_ if r["c"] <= last]).mark_line(color=color, strokeWidth=2.5).encode(x=x, y=y),
            values([r for r in law_ if r["c"] >= last]).mark_line(color=color, strokeWidth=2.5, strokeDash=[6, 4]).encode(x=x, y=y),
            values(mins).mark_circle(color=color, size=70, opacity=1).encode(x=x, y=y),
            values([dict(c=lab[fam][0], v=lab[fam][1], t=name)]).mark_text(align="left", color=color, fontWeight=600)
            .encode(x=x, y=y, text="t:N"),
        ]  # fmt: skip
    return layers


def training():
    """The big run's main-eval CE over training, the scaling law's forecast for it, and Allie 2.0 after the second
    anneal."""
    rows = sorted(
        jsonl(X / "maia3-bench/bigrun-trajectory-v2.jsonl"), key=lambda r: r["step"]
    )
    curve = [dict(t=r["tokens"] / 1e9, v=r["golden_macro"]) for r in rows]
    act, _, D = allie_n()
    forecast = law("s16")(act, D)
    t_ann = (
        curve[-1]["t"]
        + jload(R / f"pretrain/{ANN}/done.json")["tokens_processed"] / 1e9
    )
    v_ann = jload(R / f"lm-eval/{ANN}/strat-v1.json")["macro"]
    x = alt.X("t:Q", scale=alt.Scale(domain=[20, 80], nice=False), title="Training tokens (billions)",
              axis=alt.Axis(values=[20, 30, 40, 50, 60, 70, 80], grid=False))  # fmt: skip
    y = alt.Y("v:Q", scale=alt.Scale(domain=[1.22, 1.42], nice=False, zero=False), title="Main-evaluation loss (nats)",
              axis=alt.Axis(values=[1.25, 1.30, 1.35, 1.40], grid=True, format=".2f"))  # fmt: skip
    end = [dict(curve[-1], t_=f"Big run  {curve[-1]['v']:.4f}")]
    ann = [dict(t=t_ann, v=v_ann, t_=f"Allie 2.0  {v_ann:.4f}")]
    fc = [dict(t=D / 1e9, v=forecast, t_=f"Forecast  {forecast:.4f}")]
    chart = alt.layer(
        values(curve).mark_line(color=ALLIE, strokeWidth=3).encode(x=x, y=y),
        values(end + ann).mark_line(color=ALLIE, strokeWidth=2, strokeDash=[3, 2]).encode(x=x, y=y),
        values(end).mark_circle(color=ALLIE, size=60, opacity=1).encode(x=x, y=y),
        values(end).mark_text(align="right", dx=-12, dy=12).encode(x=x, y=y, text="t_:N"),
        values(ann).mark_circle(color=ALLIE, size=120, opacity=1).encode(x=x, y=y),
        values(ann).mark_text(align="right", dx=-22, dy=30, color=INK, fontWeight=600).encode(x=x, y=y, text="t_:N"),
        values(fc).mark_point(color=NEUTRAL_DARK, size=120, strokeWidth=2, filled=False).encode(x=x, y=y),
        values(fc).mark_text(dy=-18).encode(x=x, y=y, text="t_:N"),
    ).properties(width=WIDTH, height=300, title=title(
        "Allie 2.0's loss kept falling to the end of training",
        "Main-evaluation loss of the big run's checkpoints from 22B tokens on, the scaling law's forecast for it,",
        "and Allie 2.0 after a second anneal on 1B recent tokens. Lower is better."))  # fmt: skip
    save(chart, "training")
    print(
        f"  N {act / 1e6:.1f}M, D {D / 1e9:.1f}B: forecast {forecast:.4f}; big run {curve[-1]['v']:.4f} from {curve[0]['t']:.0f}B;"
        f" Allie 2.0 {v_ann:.4f} at {t_ann:.1f}B"
    )


def isoflop():
    """Isoflop curves, MoE against dense, one panel per budget: quadratic fits in log N through each budget's
    selected four-size window (as the sweep readout fits them, every run), the window's seed means and the
    fitted minima."""
    sweep, cells = jload(SWEEP), macro()["cells"]
    panels = []
    for i, b in enumerate(("1e17", "3e17", "1e18")):
        x = alt.X("n:Q", scale=alt.Scale(type="log", domain=[1.2e7, 5e8], nice=False), title="Active parameters" if i == 1 else None,
                  axis=alt.Axis(values=[3e7, 1e8, 3e8], grid=False, labelExpr="format(datum.value / 1e6, 'd') + 'M'"))  # fmt: skip
        y = alt.Y("v:Q", scale=alt.Scale(domain=[1.30, 1.46], nice=False, zero=False), title="Main-evaluation loss (nats)" if i == 0 else None,
                  axis=alt.Axis(values=[1.30, 1.35, 1.40, 1.45], grid=True, format=".2f", labels=i == 0))  # fmt: skip
        layers = []
        for fam, color, name in (
            ("dense", NEUTRAL_DARK, "dense"),
            ("s16", ALLIE, "MoE"),
        ):
            cell = cells[fam][b]
            runs = [
                r
                for r in sweep["runs"]
                if r["fam"] == fam and r["budget"] == b and r["shape"] in cell["fit"]
            ]
            n, v = (
                np.array([r["n"] for r in runs]),
                np.array([r["y"]["macro"] for r in runs]),
            )
            c = np.polyfit(np.log(n), v, 2)
            xs = np.linspace(np.log(n).min() - 0.15, np.log(n).max() + 0.15, 60)
            m = -c[1] / (2 * c[0])
            opt = dict(n=math.exp(m), v=float(np.polyval(c, m)), t=name)
            layers += [
                values([dict(n=float(math.exp(t)), v=float(np.polyval(c, t))) for t in xs]).mark_line(color=color, strokeWidth=2.5).encode(x=x, y=y),
                values([dict(n=float(k), v=float(v[n == k].mean())) for k in np.unique(n)]).mark_circle(color=color, size=40, opacity=1).encode(x=x, y=y),
                values([opt]).mark_point(color=color, size=170, strokeWidth=2).encode(x=x, y=y),
            ]  # fmt: skip
            if i == 0:
                layers.append(
                    values([opt])
                    .mark_text(
                        dy=20 if fam == "s16" else -18, color=color, fontWeight=600
                    )
                    .encode(x=x, y=y, text="t:N")
                )
            print(f"  {b} {fam:5s} N* {opt['n'] / 1e6:6.1f}M L* {opt['v']:.4f}")
        c = cells["s16"][b]["c"]
        e = 10 ** math.floor(math.log10(c))
        panels.append(alt.layer(*layers).properties(width=202, height=250, title=alt.TitleParams(
            f"{c / e:.1f} × {power(e)} FLOPs", anchor="start", fontSize=13, fontWeight=600, color=INK, offset=8)))  # fmt: skip
    chart = alt.hconcat(*panels, spacing=14).properties(title=title(
        "The mixture of experts beats dense models at every budget",
        "Main-evaluation loss against model size at three training budgets. Rings mark the best size."))  # fmt: skip
    save(chart, "isoflop")


def moe_vs_dense():
    """Dense compute needed to match the MoE, as a ratio, from the two fitted laws, with 90% noise-bootstrap intervals."""
    cm = macro()["cm"]["s16"]
    rows = [dict(c=cm[b]["c"], v=cm[b]["separate"]["cm"], lo=cm[b]["separate"]["noise"][0], hi=cm[b]["separate"]["noise"][1],
                 ext=b == "1e19 FLOPs") for b in ("1e17", "3e17", "1e18", "1e19 FLOPs")]  # fmt: skip
    for r in rows:
        r["t"] = f"{r['v']:.1f}×" + (" (extrapolated)" if r["ext"] else "")
    sx, sy = (
        alt.Scale(type="log", domain=[4e17, 2e19], nice=False),
        alt.Scale(domain=[0.8, 4.0], nice=False),
    )
    x = alt.X(
        "c:Q",
        scale=sx,
        title="Training compute (FLOPs)",
        axis=alt.Axis(values=[1e18, 1e19], grid=False, labelExpr=power_axis()),
    )
    y = alt.Y(
        "v:Q",
        scale=sy,
        title=None,
        axis=alt.Axis(values=[1, 2, 3, 4], grid=True, labelExpr="datum.value + '×'"),
    )
    b = values(rows)
    chart = alt.layer(
        values([{"v": 1}]).mark_rule(color=NEUTRAL_DARK, strokeWidth=1.5).encode(y=y),
        values([{"v": 1, "c": 1.6e19, "t": "dense"}]).mark_text(dy=-9, color=MUTED).encode(x=x, y=y, text="t:N"),
        b.mark_rule(color=INK2, strokeWidth=1.2).encode(x=x, y=alt.Y("lo:Q", scale=sy), y2="hi:Q"),
        b.mark_circle(size=150, opacity=1).encode(x=x, y=y, color=alt.condition(alt.datum.ext, alt.value(ALLIE_PALE), alt.value(ALLIE))),
        b.mark_text(align="left", dx=12, color=INK, fontWeight=600).encode(x=x, y=y, text="t:N"),
    ).properties(width=WIDTH + 6, height=260, title=title(
        "A dense model needs over twice the compute to match the mixture of experts",
        "Compute a dense model needs to reach the mixture of experts' loss, as a multiple. Whiskers: 90%",
        "intervals. The last point extrapolates beyond the sweep."))  # fmt: skip
    save(chart, "moe-vs-dense")
    for r in rows:
        print(f"  {r['c']:.2e}: {r['v']:.2f}x [{r['lo']:.2f}, {r['hi']:.2f}]")


def frontier():
    """Main-eval CE against training compute: the isoflop minima, each family's law at its compute-optimal size,
    and the big run against the law's forecast for its own size and tokens."""
    act, _, D = allie_n()
    final = jload(R / f"lm-eval/{FINAL.name}/strat-v1.json")["macro"]
    forecast, E = law("s16")(act, D), math.exp(macro()["law"]["s16"]["theta"][0])
    sx, sy = (
        alt.Scale(type="log", domain=[4e17, 3e21], nice=False),
        alt.Scale(domain=[1.24, 1.42], nice=False, zero=False),
    )
    x = alt.X(
        "c:Q",
        scale=sx,
        title="Training compute (FLOPs)",
        axis=alt.Axis(
            values=[1e18, 1e19, 1e20, 1e21], grid=False, labelExpr=power_axis()
        ),
    )
    y = alt.Y(
        "v:Q",
        scale=sy,
        title="Main-evaluation loss (nats)",
        axis=alt.Axis(values=[1.25, 1.30, 1.35, 1.40], grid=True, format=".2f"),
    )
    chart = alt.layer(
        *law_layers(x, y, 0, {"dense": (2.3e18, 1.374), "s16": (6.5e17, 1.311)}),
        values([{"v": E}]).mark_rule(color=NEUTRAL, strokeDash=[2, 3], strokeWidth=1.5).encode(y=y),
        values([{"c": 5e17, "v": E, "t": "the law's floor"}]).mark_text(align="left", dy=-9, color=MUTED).encode(x=x, y=y, text="t:N"),
        values([{"c": C_ALLIE, "v": final, "v2": forecast}]).mark_rule(color=NEUTRAL, strokeWidth=1.5).encode(x=x, y=y, y2="v2:Q"),
        values([{"c": C_ALLIE, "v": forecast}]).mark_point(color=ALLIE, size=150, strokeWidth=2, filled=False).encode(x=x, y=y),
        values([{"c": C_ALLIE, "v": forecast, "t": f"Forecast  {forecast:.4f}"}]).mark_text(align="left", dx=12, dy=-4).encode(x=x, y=y, text="t:N"),
        values([{"c": C_ALLIE, "v": final}]).mark_circle(color=ALLIE, size=170, opacity=1).encode(x=x, y=y),
        values([{"c": C_ALLIE, "v": final, "t": f"Big run  {final:.4f}"}]).mark_text(align="left", dx=12, color=INK, fontWeight=600).encode(x=x, y=y, text="t:N"),
    ).properties(width=WIDTH, height=300, title=title(
        "The big run landed below the scaling law's forecast",
        "Best main-evaluation loss at each training budget. Dashed: the law's extrapolation. The big run landed",
        "0.032 below its forecast, and below the law's floor."))  # fmt: skip
    save(chart, "frontier")
    l_opt, n_opt, d_opt = optimum("s16", C_ALLIE)
    print(
        f"  big run {C_ALLIE:.3e} FLOPs: {final:.4f}, forecast {forecast:.4f}, law optimum {l_opt:.4f}"
    )
    print(
        f"  law optimum at that compute: N* {n_opt / 1e9:.2f}B, D* {d_opt / 1e9:.1f}B; floor E {E:.4f}"
    )


def optimal():
    """Compute-optimal active parameters against training compute: the isoflop minima, each law's optimum, and
    the big run's size against the law's extrapolated choice at its compute."""
    act, _, D = allie_n()
    _, n_opt, d_opt = optimum("s16", C_ALLIE)
    sx, sy = (
        alt.Scale(type="log", domain=[4e17, 3e21], nice=False),
        alt.Scale(type="log", domain=[1.5e7, 5e9], nice=False),
    )
    label = "datum.value >= 1e9 ? format(datum.value / 1e9, '~g') + 'B' : format(datum.value / 1e6, '~g') + 'M'"
    x = alt.X(
        "c:Q",
        scale=sx,
        title="Training compute (FLOPs)",
        axis=alt.Axis(
            values=[1e18, 1e19, 1e20, 1e21], grid=False, labelExpr=power_axis()
        ),
    )
    y = alt.Y(
        "v:Q",
        scale=sy,
        title="Active parameters",
        axis=alt.Axis(values=[3e7, 1e8, 3e8, 1e9, 3e9], grid=True, labelExpr=label),
    )
    chart = alt.layer(
        *law_layers(x, y, 1, {"dense": (9e17, 1.6e8), "s16": (2.5e18, 4.5e7)}),
        values([{"c": C_ALLIE, "v": n_opt}]).mark_point(color=ALLIE, size=150, strokeWidth=2, filled=False).encode(x=x, y=y),
        values([{"c": C_ALLIE, "v": n_opt, "t": ["Extrapolated optimum", f"{n_opt / 1e9:.1f}B, {d_opt / 1e9:.0f}B tokens"]}])
        .mark_text(align="left", dx=14, lineHeight=17).encode(x=x, y=y, text="t:N"),
        values([{"c": C_ALLIE, "v": act}]).mark_circle(color=ALLIE, size=170, opacity=1).encode(x=x, y=y),
        values([{"c": C_ALLIE, "v": act, "t": ["Big run", f"{act / 1e9:.2f}B, {D / 1e9:.0f}B tokens"]}])
        .mark_text(align="left", dx=14, lineHeight=17, color=INK, fontWeight=600).encode(x=x, y=y, text="t:N"),
    ).properties(width=WIDTH, height=300, title=title(
        "The law would have picked a larger model on fewer tokens",
        "Compute-optimal model size: the sweep's best sizes and each law's optimum, dashed beyond the sweep.",
        "The extrapolation reaches 51 times past the sweep's largest budget."))  # fmt: skip
    save(chart, "optimal")
    print(
        f"  law optimum at {C_ALLIE:.3e}: {n_opt / 1e9:.2f}B on {d_opt / 1e9:.1f}B; big run {act / 1e9:.3f}B"
    )


# ------------------------------------------------------------------------------------ the model


def box(ax, x, y, w, h, text="", ec=NEUTRAL, fc="white", size=12.5, color=INK, lw=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=2,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            zorder=3,
        )


def arrow(ax, a, b, color=NEUTRAL):
    style = {"arrowstyle": "-|>", "mutation_scale": 12, "color": color, "lw": 1.3}
    ax.add_patch(FancyArrowPatch(a, b, shrinkA=0, shrinkB=0, zorder=4, **style))


def model():
    """Inputs, one mixture-of-experts block (schematic), the next-move output; coordinates in inches."""
    for f in FONTS.glob("Inter-*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": ["Inter"],
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 7.2), ylim=(0, 3.6))
    ax.axis("off")
    mid = 1.7
    small = {
        "ha": "center",
        "va": "center",
        "fontsize": 11.5,
        "color": INK2,
        "zorder": 4,
    }
    for y, text in ((2.55, "Game +\nratings"), (1.7, "Clock"), (0.85, "Board")):
        box(ax, 0.1, y - 0.3, 1.0, 0.6, text)
        arrow(ax, (1.1, y), (1.4, mid + (y - mid) * 0.35))
    box(ax, 1.4, 0.2, 4.0, 3.2, ec="#ddd6c8", fc="#f1ede4")
    ax.text(3.4, 3.15, "MoE block  (23 of 24 blocks)", **small)
    box(ax, 1.55, mid - 0.3, 1.15, 0.6, "Attention")
    ax.plot(
        [2.7, 2.95, 2.95, 3.75], [mid, mid, 2.6, 2.6], color=ALLIE, lw=1.3, zorder=3
    )  # the shared expert
    box(ax, 3.75, 2.45, 1.05, 0.3, ec=ALLIE, fc=ALLIE_WASH)
    ax.text(4.275, 2.82, "shared expert", **small | {"va": "bottom"})
    ax.plot([4.8, 5.15, 5.15], [2.6, 2.6, mid + 0.09], color=ALLIE, lw=1.3, zorder=3)
    ax.plot(
        [2.7, 3.25], [mid, mid], color=MUTED, lw=1.3, zorder=3
    )  # the router and its routed experts
    ax.add_patch(plt.Circle((3.3, mid), 0.07, fc=MUTED, lw=0, zorder=5))
    ax.text(3.0, mid - 0.12, "router", **small | {"va": "top", "color": MUTED})
    for on, y in zip(
        [False, True, None, False, True, False], np.linspace(2.05, 0.75, 6)
    ):
        if on is None:
            ax.text(4.275, y, "···", **small | {"fontsize": 14, "color": MUTED})
            continue
        box(
            ax,
            3.75,
            y - 0.09,
            1.05,
            0.18,
            ec=ALLIE if on else NEUTRAL,
            fc=ALLIE_WASH if on else "#fbfaf7",
        )
        color, z = (ALLIE, 3) if on else ("#e4e0d8", 2.5)
        ax.plot([3.3, 3.75], [mid, y], color=color, lw=1.3, zorder=z)
        ax.plot([4.8, 5.08], [y, mid], color=color, lw=1.3, zorder=z)
    ax.text(4.275, 0.42, "16 of 256 routed experts", **small)
    ax.add_patch(plt.Circle((5.15, mid), 0.09, fc="white", ec=MUTED, lw=1.2, zorder=5))
    ax.text(5.15, mid, "+", **small | {"fontsize": 10, "color": MUTED, "zorder": 6})
    arrow(ax, (5.24, mid), (5.7, mid), color=MUTED)
    box(ax, 5.7, mid - 0.4, 1.4, 0.8, "Next-move\ndistribution", ec=ALLIE, lw=1.6)
    ax.text(
        6.4,
        mid - 0.55,
        "also: time spent,\ngame result",
        **small | {"va": "top", "color": MUTED, "linespacing": 1.2},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "model.png", dpi=200)
    plt.close(fig)
    print("wrote docs/figures/model.png")


# -------------------------------------------------------------------- comparison with earlier models


def previous(name, positions):
    """The original Allie's raw-policy legal-move CE per position on a set (bench: the 80,000 benchmark positions,
    rating: every scored blitz move), in that set's order; None if not scored."""
    f = ORIGINAL / positions / f"{name}.npz"
    if not f.exists():
        print(f"  (no scores at {f})")
        return None
    with np.load(f) as z:
        return np.asarray(z["ce_legal"], float)


def gflops(name):
    """The original Allie's GFLOPs per move with a key-value cache, counted as for the other models."""
    return (
        jload(ORIGINAL / f"flops-{name.split('-')[1]}.json")[
            "flops_per_move_incremental"
        ]
        / 1e9
    )


def search_index(n):
    """Benchmark positions the search ran on (legal.npz order), as run.py drew them."""
    files = sorted((SEARCH / "ann2/legal").glob("[0-9]*.npz"))
    return np.concatenate([np.load(f)["index"] for f in files])[:n]


def pareto():
    """Legal-move CE against GFLOPs per move, on the 20,000 benchmark positions that search ran on."""
    rep = jload(SEARCH / "report-ann2.json")
    p = rep["points"]
    pts = [
        dict(fam="maia", g=p[f"maia/{m}"]["gflops"], v=p[f"maia/{m}"]["ce"], t=name)
        for m, name in MAIA.items()
    ]
    for k, t in (("frozen/legal", "Allie 2.0 (raw)"), ("devcal/5", "5 simulations")):
        pts.append(dict(fam="allie", g=p[k]["gflops"], v=p[k]["ce"], t=t))
    ce = previous("allie1-medium", "bench")
    if ce is not None:
        pts.append(
            dict(
                fam="orig",
                g=gflops("allie1-medium"),
                v=float(ce[search_index(rep["n"])].mean()),
                t="Original Allie",
            )
        )
    x = alt.X("g:Q", scale=alt.Scale(type="log", domain=[0.2, 25], nice=False), title="Compute per move (GFLOPs, log scale)",
              axis=alt.Axis(values=[0.5, 1, 2, 5, 10, 20], grid=False, format="~g"))  # fmt: skip
    y = alt.Y("v:Q", scale=alt.Scale(domain=[1.20, 1.32], nice=False, zero=False), title="Loss (nats, lower is better)",
              axis=alt.Axis(values=[1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32], grid=True, format=".2f"))  # fmt: skip
    b = values(pts)
    maia, allie, orig = (
        b.transform_filter(alt.datum.fam == f) for f in ("maia", "allie", "orig")
    )
    raw = alt.datum.t == "Allie 2.0 (raw)"
    chart = alt.layer(
        maia.mark_line(color=NEUTRAL_DARK, strokeWidth=2).encode(x=x, y=y),
        maia.mark_circle(size=110, color=NEUTRAL_DARK, opacity=1).encode(x=x, y=y),
        maia.mark_text(align="left", dx=10, dy=-10).encode(x=x, y=y, text="t:N"),
        orig.mark_point(size=110, color=NEUTRAL_DARK, strokeWidth=2, filled=False).encode(x=x, y=y),
        orig.mark_text(align="right", dx=-11).encode(x=x, y=y, text="t:N"),
        allie.mark_line(color=ALLIE, strokeWidth=2.4).encode(x=x, y=y),
        allie.mark_circle(color=ALLIE, opacity=1).encode(x=x, y=y, size=alt.condition(raw, alt.value(170), alt.value(90))),
        allie.transform_filter(raw).mark_text(dy=20, color=INK, fontWeight=600).encode(x=x, y=y, text="t:N"),
        allie.transform_filter(~raw).mark_text(dy=18).encode(x=x, y=y, text="t:N"),
    ).properties(width=2 * 300 + 34, height=330, title=title(
        "Allie 2.0 needs 1.4 GFLOPs per move",
        "Loss against compute per move on 20,000 blitz positions, every position scored. Compute counts",
        "matrix multiplies. Grey line: a guide between three separately trained Maia-3 models."))  # fmt: skip
    save(chart, "pareto")
    for r in pts:
        print(f"  {r['t']:16s} {r['g']:7.2f} GF  CE {r['v']:.4f}")


def protocol_scores():
    """Each model's benchmark scores per position, in games.npz's sampled order: (legal-move CE, top-1)."""
    with np.load(DATA / "maia3-bench/games.npz") as z:
        keep = z["keep"]

    def read(path):
        with np.load(path) as z:
            ce, top1 = (
                (z["ce_legal"], z["top1"]) if "ce_legal" in z else (z["ce"], z["top1"])
            )
        return (ce, top1) if len(ce) == len(keep) else (ce[keep], top1[keep])

    paths = {name: DATA / f"maia3-bench/scores/{m}.npz" for m, name in MAIA.items()}
    paths["Original Allie"] = ORIGINAL / "bench/allie1-medium.npz"
    paths["Allie 2.0"] = DATA / f"distill-v2/scores/{ANN}.npz"
    return {k: read(p) for k, p in paths.items() if Path(p).exists()}


def protocol():
    """On Maia-3's evaluation rule: each model's loss and top-1 accuracy minus Maia-3 79M's, paired on the same
    positions, with 95% game-bootstrap intervals (allie.eval.maia3.aggregate.compare)."""
    sys.path[:0] = [str(ROOT / "src")]
    from allie.eval.maia3 import aggregate

    res = aggregate.compare(protocol_scores())["maia_protocol"]
    order = [
        m
        for m in ("Allie 2.0", "Maia-3 23M", "Original Allie", "Maia-3 5M")
        if m in res
    ]
    panels = []
    for i, (key, name, dom, nd, ticks, lnd) in enumerate((
        ("d_ce", "Loss difference (nats, lower is better)", [-0.04, 0.10], 2, [-0.04, 0, 0.04, 0.08], 3),
        ("d_top1", "Top-choice accuracy (percentage points)", [-3.0, 1.3], 0, [-3, -2, -1, 0, 1], 2),
    )):  # fmt: skip
        rows = [dict(name=m, d=res[m][key]["mean"], lo=res[m][key]["lo"], hi=res[m][key]["hi"], label=signed(res[m][key]["mean"], lnd))
                for m in order]  # fmt: skip
        panels.append(
            diverging(
                rows,
                order,
                dom,
                nd,
                ticks,
                name,
                alt.datum.name == "Allie 2.0",
                label_axis=i == 0,
                width=280,
            )
        )
    chart = alt.hconcat(*panels, spacing=34).properties(title=title(
        "Under the Maia-3 protocol, Allie 2.0 performs similarly to Maia-3 79M",
        f"Difference from Maia-3 79M on {res['positions']:,} blitz positions, scored by the Maia-3 paper's rule.",
        "Whiskers: 95% intervals."))  # fmt: skip
    save(chart, "protocol")
    print(f"  {res['positions']} positions")
    for n in order + ["Maia-3 79M"]:
        r = res[n]
        d = (
            f"  dCE {signed(r['d_ce']['mean'], 4)}  dtop1 {signed(r['d_top1']['mean'], 2)}"
            if "d_ce" in r
            else ""
        )
        print(f"  {n:16s} CE {r['ce']:.4f} top-1 {r['top1']:.2f}{d}")


def versatility():
    """Allie 2.0's loss minus Maia-3 79M's by time control (bullet, rapid and classical from the formats sample,
    5,000 positions per rating band; blitz from the benchmark) and, in blitz, by time left on the mover's clock;
    paired 95% game-bootstrap intervals (allie.eval.maia3.report's aggregate-NAME.json)."""
    sl = jload(X / f"maia3-bench/aggregate-{ANN}.json")["slices"]
    clocks = ["Under 10 s", "10–30 s", "30–60 s", "1–2 min", "Over 2 min"]
    groups = (
        ("By time control", ["bullet", "blitz", "rapid", "classical"], [sl["format"][f] for f in ("bullet", "blitz", "rapid", "classical")]),
        ("Blitz, by time left on the clock", clocks, list(sl["clock"].values())),
    )  # fmt: skip
    panels = []
    for name, order, items in groups:
        rows = [dict(name=n, d=d["d_ce_maia3-79m"]["mean"], lo=d["d_ce_maia3-79m"]["lo"], hi=d["d_ce_maia3-79m"]["hi"],
                     label=signed(d["d_ce_maia3-79m"]["mean"], 3)) for n, d in zip(order, items)]  # fmt: skip
        panels.append(diverging(rows, order, [-0.37, 0.06], 1, [-0.3, -0.2, -0.1, 0], name, alt.datum.d == alt.datum.d, width=248)
                      .properties(height=34 * 5))  # fmt: skip
        for r in rows:
            print(
                f"  {r['name']:11s} dCE {signed(r['d'], 4)} [{signed(r['lo'], 4)}, {signed(r['hi'], 4)}]"
            )
    chart = alt.hconcat(*panels, spacing=34).properties(title=title(
        "Allie 2.0's lead is largest in bullet and in time trouble",
        "Allie 2.0's loss minus Maia-3 79M's; below zero, Allie 2.0 predicts better. Maia-3 never",
        "trained on bullet, rapid or classical games. Whiskers: 95% intervals."))  # fmt: skip
    save(chart, "versatility")


def rating():
    """Legal-move CE minus Maia-3 79M's per 100-point bin of game rating, on every scored blitz move."""
    rows = [
        r for r in jload(X / f"bigrun-progress/acc-by-game-rating-{ANN}.json") if r["n"]
    ]
    x_ = [float(np.mean([float(v) for v in r["bin"].split("-")])) for r in rows]
    ref = np.array([r["models"]["maia3-79m"]["ce"][0] for r in rows])
    pts = []
    for i, r in enumerate(rows):
        d = r["models"][ANN]["d_ce_79m"]
        pts.append(dict(x=x_[i], s="Allie 2.0", d=d[0], lo=d[1], hi=d[2]))
        pts += [
            dict(x=x_[i], s=MAIA[m], d=r["models"][m]["ce"][0] - ref[i])
            for m in ("maia3-5m", "maia3-23m")
        ]
    ce = previous("allie1-medium", "rating")
    if ce is not None:
        pts += [
            dict(x=x, s="Original Allie", d=float(v))
            for x, v in zip(x_, binned(ce) - ref)
        ]
    sx, sy = (
        alt.Scale(domain=[600, 2900], nice=False),
        alt.Scale(domain=[-0.11, 0.16], nice=False),
    )
    x = alt.X(
        "x:Q",
        scale=sx,
        title="Game rating (Lichess blitz)",
        axis=alt.Axis(
            values=[800, 1200, 1600, 2000, 2400, 2800], grid=False, format="d"
        ),
    )
    y = alt.Y(
        "d:Q",
        scale=sy,
        title="Loss minus Maia-3 79M's (nats)",
        axis=alt.Axis(
            values=[-0.1, -0.05, 0, 0.05, 0.1, 0.15],
            grid=True,
            labelExpr=signed_axis(2),
        ),
    )
    b = values(pts)
    allie = b.transform_filter(alt.datum.s == "Allie 2.0")
    layers = [values([{"d": 0}]).mark_rule(color=INK, strokeWidth=1.4).encode(y=y),
              allie.mark_area(color=ALLIE, opacity=0.15).encode(x=x, y=alt.Y("lo:Q", scale=sy), y2="hi:Q")]  # fmt: skip
    ends = {p["s"]: p["d"] for p in pts}
    labels = [
        dict(x=2850, d=-0.016, t="Maia-3 79M"),
        dict(x=1950, d=-0.04, t="Allie 2.0"),
    ]
    for s, color, dash, dy in (
        ("Maia-3 5M", NEUTRAL, [1, 0], -0.004),
        ("Maia-3 23M", NEUTRAL_DARK, [1, 0], 0),
        ("Original Allie", INK2, [5, 3], 0.004),
    ):
        if s in ends:
            layers.append(
                b.transform_filter(alt.datum.s == s)
                .mark_line(color=color, strokeWidth=2, strokeDash=dash)
                .encode(x=x, y=y)
            )
            labels.append(dict(x=2850, d=ends[s] + dy, t=s))
    layers.append(allie.mark_line(color=ALLIE, strokeWidth=3).encode(x=x, y=y))
    lab = values(labels)
    layers += [lab.transform_filter(alt.datum.t != "Allie 2.0").mark_text(align="left", dx=10).encode(x=x, y=y, text="t:N"),
               lab.transform_filter(alt.datum.t == "Allie 2.0").mark_text(align="left", color=ALLIE, fontWeight=700).encode(x=x, y=y, text="t:N")]  # fmt: skip
    chart = alt.layer(*layers).properties(width=546, height=300, title=title(
        "Allie 2.0 predicts well at every rating",
        "Loss minus Maia-3 79M's by game rating, on all blitz positions of the main evaluation. Below zero is",
        "better. Band: Allie 2.0's 95% interval; the end bins hold few games."))  # fmt: skip
    save(chart, "rating")
    d = np.array([r["models"][ANN]["d_ce_79m"] for r in rows])
    print(
        f"  {len(rows)} bins; CE above 79M in {[r['bin'] for r, v in zip(rows, d[:, 0]) if v > 0]}; interval below 0 in {(d[:, 2] < 0).sum()}"
    )


def calibrated():
    """Move accuracy and blunder rate (Stockfish on every legal move) against rating, per time control: the humans'
    own moves (line, 95% band), and the expected values of the move distributions the Lichess bot plays at that
    rating, sampled straight from the policy or calibrated as shipped (allie.lichess.calibration: per position,
    its search rungs mixed by the think times the bot draws), with 95% intervals clustered by game
    (analysis/elo_strength/calib_unified.py --params --plot)."""
    data = jload(X / "elo-strength/calib/plot-unified.json")
    formats = ("bullet", "blitz", "rapid", "classical")
    who = ("Humans", "Allie (raw policy)", "Allie (calibrated)")
    pts, n = [], 0
    for key, c in data["cells"].items():
        f, b = key.split("/")
        if f not in formats or not 800 <= int(b) <= 2600:
            continue
        n += c["n"]
        for w, src, dx in ((who[0], "human", 0), (who[1], "raw", -30), (who[2], "calibrated", 30)):
            for metric, scale in (("accuracy", 1), ("blunder", 100)):
                v, e = c[src][metric] * scale, 1.96 * c[src][metric + "_se"] * scale
                pts.append({"f": f, "x": int(b) + dx, "s": w, "m": metric, "v": v, "lo": v - e, "hi": v + e})  # fmt: skip
    color = alt.Color("s:N", title=None, scale=alt.Scale(domain=list(who), range=[INK2, NEUTRAL_DARK, ALLIE]),
                      legend=alt.Legend(orient="top", direction="horizontal", labelFontSize=15, symbolSize=160,
                                        symbolStrokeWidth=2.5, columnPadding=28, offset=12))  # fmt: skip
    shape = alt.Shape("s:N", scale=alt.Scale(domain=list(who), range=["stroke", "circle", "diamond"]), legend=None)  # fmt: skip
    big = {"labelFontSize": 14, "titleFontSize": 15, "titleFontWeight": 600, "titleColor": INK}  # fmt: skip
    rows = []
    for r, (metric, label, dom, log, ticks) in enumerate((
        ("accuracy", "Move accuracy", [83, 96.6], False, [84, 87, 90, 93, 96]),
        ("blunder", "Blunder rate", [1, 14], True, [1, 2, 5, 10]),
    )):  # fmt: skip
        panels = []
        for i, f in enumerate(formats):
            b = values([p for p in pts if p["f"] == f and p["m"] == metric])
            sy = alt.Scale(domain=dom, type="log" if log else "linear", nice=False)
            x = alt.X("x:Q", scale=alt.Scale(domain=[650, 2750], nice=False), title="Rating" if r == 1 else None,
                      axis=alt.Axis(values=[800, 1700, 2600], grid=False, format="d", **big))  # fmt: skip
            y = alt.Y("v:Q", scale=sy, title=label if i == 0 else None,
                      axis=alt.Axis(values=ticks, grid=True, labels=i == 0, labelExpr="datum.value + '%'", **big))  # fmt: skip
            hum = b.transform_filter(alt.datum.s == who[0])
            dots = b.transform_filter(alt.datum.s != who[0])
            layers = [
                hum.mark_area(color=INK2, opacity=0.12, clip=True).encode(x=x, y=alt.Y("lo:Q", scale=sy), y2="hi:Q"),
                hum.mark_line(strokeWidth=2.5).encode(x=x, y=y, color=color),
                dots.mark_rule(strokeWidth=1.4, clip=True).encode(x=x, y=alt.Y("lo:Q", scale=sy), y2="hi:Q", color=color),
                dots.mark_point(filled=True, size=70, opacity=1).encode(x=x, y=y, color=color, shape=shape),
            ]  # fmt: skip
            panel = alt.layer(*layers).properties(width=150, height=180)
            t = alt.TitleParams(f.capitalize(), anchor="middle", fontSize=16, fontWeight=600, color=INK, offset=8)  # fmt: skip
            panels.append(panel.properties(title=t) if r == 0 else panel)
        rows.append(alt.hconcat(*panels, spacing=12))
    chart = alt.vconcat(*rows, spacing=14).properties(title=title("Calibrated play tracks human move quality"))  # fmt: skip
    save(chart, "calibrated")
    print(f"  {n:,} positions")
    for f in formats:
        sl = data["slopes"][f]
        print(f"  {f:9s} slope vs humans: accuracy policy {sl['accuracy']['raw']['ratio']:.2f}x, calibrated "
              f"{sl['accuracy']['calibrated']['ratio']:.2f}x; blunder policy {sl['blunder']['raw']['ratio']:.2f}x, "
              f"calibrated {sl['blunder']['calibrated']['ratio']:.2f}x")  # fmt: skip


def binned(values_):
    """values_ (rating-set order) averaged per 100-point game-rating bin, weighted to the natural mover mix."""
    with np.load(DATA / "maia3-bench/rating/games.npz") as z:
        sel, meta = z["sel"][z["keep"]], z["meta"]
    m = jload(DATA / "strat-eval-v1/manifest.json")
    w = (np.array(m["population_moves"]) / np.array(m["scored_moves"]))[sel[:, 2]]
    rating = meta[sel[:, 0], 2:4].mean(1)
    edges = np.arange(600, 2901, 100)
    keep = [(rating >= lo) & (rating < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return np.array([np.average(values_[k], weights=w[k]) for k in keep if k.any()])


FIGS = {
    "model": model,
    "isoflop": isoflop,
    "moe-vs-dense": moe_vs_dense,
    "frontier": frontier,
    "optimal": optimal,
    "training": training,
    "pareto": pareto,
    "protocol": protocol,
    "versatility": versatility,
    "rating": rating,
    "calibrated": calibrated,
}

if __name__ == "__main__":
    plt.switch_backend("agg")
    vlc.register_font_directory(str(FONTS))
    for name in sys.argv[1:] or FIGS:
        print(name)
        FIGS[name]()
