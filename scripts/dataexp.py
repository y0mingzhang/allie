"""Data-mixing waves: ours trained with --mix, one single-GPU run per (policy, seed, size).

plan WAVE     freeze sources (with chessmix) and evaluator, write plan.json and the sbatch
submit WAVE   submit the array once
task WAVE     run one array task: train with --mix, score on original validation, publish result
status WAVE   one line per run against the isoflop-v1 baseline of its size
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/yimingz3/src/allie")
DATA = "/data/group_data/dei-group/yimingz3/allie/lichess_tokens_v2"  # validation rows only
OLD_EVAL = ROOT / "results/recipe10x/isoflop-v1/evaluator-ours"
SCHEDULE = dict(
    warmup_steps=32,
    mtp_steps=64,
    split_step=65,
    batch_rows=512,
    plateau=4.0,
    final_lr=0.2,
    decay_shape="linear",
)
POLICIES = [
    "natural",
    "mover_rule",
    "up2",
    "up4",
    "bots_masked",
    "cooldown20",
    "less_bullet",
]
# compute-optimal isoflop-v1 shapes and their packed-corpus scores (move, >=2400)
SIZES = {
    # 1e16: law optimum ~13.5M non-embedding params; 8 layers is the minimum depth
    "1e16": dict(layers=8, width=384, steps=1347, base=(float("nan"), float("nan"))),
    "3e16": dict(layers=8, width=512, steps=2274, base=(1.5664, 1.4971)),
    "1e17": dict(layers=12, width=512, steps=5053, base=(1.4861, 1.3871)),
    "3e17": dict(layers=16, width=768, steps=5053, base=(1.4376, 1.3174)),
}


def runs(budget, policies, seeds=(42,)):
    return [dict(policy=p, seed=s, budget=budget) for p in policies for s in seeds]


WAVES = dict(
    wave2=dict(
        study="data-v1-wave2",
        prefix="dmix2",
        runs=runs("1e17", ["control"], (42, 43, 44))
        + runs("1e17", POLICIES)
        + runs("3e17", ["control"]),
        sbatch="""#SBATCH --account=dippolit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-x9-32
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00""",
        purpose="transfer screen of every wave-1 policy at 1e17, plus the 3e17 control for the confirm stage",
    ),
)


WAVES["wave3"] = WAVES["wave2"] | dict(
    study="data-v1-wave3",
    prefix="dmix3",
    runs=runs("3e17", ["control"], (43, 44)) + runs("3e17", POLICIES),
    purpose="3e17 point of the scan for every policy, completing 3e16/1e17/3e17 for all",
)


STORES = [
    "/data/group_data/dei-group/yimingz3/allie/data-v1",
    "/data/group_data/dei-group/yimingz3/allie/data-v1-hist",
]
W4 = [
    "up4",
    "up8",
    "cooldown20",
    "cooldown20_up8",
    "cooldown40",
    "up2_cooldown20",
    "up4_cooldown20_up16",
    "balanced_c10",
    "balanced_c30",
    "cooldown20_balanced",
    "up4_balanced",
    "mover_up4",
    "slow_up4",
    "expert_slow10",
]


CURRENT_STORES = STORES[
    :1
]  # switch to STORES once data-v1-hist (2023–2024) is fully built


HISTORY = ROOT / "results/recipe10x/data-v1-history-counts.json"
B2 = ROOT / "results/recipe10x/data-v1-B2.json"


def b2_pin(w):
    """Verified sha256 of the published B_2 for waves built on it (the model track pools on it)."""
    if not w.get("b2"):
        return {}
    b2 = json.loads(B2.read_text())
    sha = b2.pop("sha256")
    digest = hashlib.sha256(json.dumps(b2, sort_keys=True).encode()).hexdigest()
    assert sha == digest, "B_2 edited after publishing"
    keys = (
        "policy",
        "feats",
        "input_lr",
        "aux_time",
        "aux_wdl",
        "history",
        "stores",
        "months",
    )
    exact = [r for r in w["runs"] if all(r.get(k) == b2.get(k) for k in keys)]
    assert exact or "history_counts" in w, "b2 wave without a run equal to B_2"
    if "history_counts" not in w:
        hist = hashlib.sha256(HISTORY.read_bytes()).hexdigest()
        assert hist == b2["history_counts_sha256"], "history counts differ from B_2's"
    assert all(set(b2["months"]) <= set(r["months"]) for r in w["runs"]), (
        "a run drops B_2 months"
    )
    return dict(b2_sha256=sha)


def screen(rs, pool_frac=0.014, history=False, stores=None, months=None, suffix=""):
    """Repetition matched: pool_frac = screen tokens / final tokens. With history, per-bucket
    repetition is that of the final run over all Lichess history (tag suffix h). months pins the
    month list so runs launched at different times see the same data."""
    return [
        r
        | dict(
            stores=stores or CURRENT_STORES,
            pool_frac=pool_frac,
            tag=f"pf{round(pool_frac * 1000):03d}" + "h" * history + suffix,
        )
        | (dict(history=True) if history else {})
        | (dict(months=list(months)) if months else {})
        for r in rs
    ]


WAVES["wave4"] = WAVES["wave2"] | dict(
    study="data-v1-wave4",
    prefix="dmix4",
    runs=screen(runs("3e16", ["control"], (42, 43, 44)) + runs("3e16", W4)),
    purpose="3e16 screen on data-v1 (2025-01..2026-08 minus 2026-07; later waves add 2023–2024), "
    "repetition matched to the 85B-token final run: stronger and longer expert cooldowns, cell "
    "balancing toward the golden eval, slow-format and combined boosts; winners go to 1e17",
)


WAVES["wave5"] = WAVES["wave2"] | dict(
    study="data-v1-wave5",
    prefix="dmix5",
    # dei A6000s, 4 runs per 4-GPU job: 7 runs take 2 of the 10 dei job slots
    pack=4,
    gpus=4,
    sbatch="""#SBATCH --account=dippolit
#SBATCH --partition=dei-group
#SBATCH --qos=dei_group_qos
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00""",
    runs=screen(
        runs("3e16", ["control"], (42, 43))
        + runs(
            "3e16",
            ["up4_2200", "up4_2600", "clean_short", "clean_term", "provisional_down"],
        )
    ),
    purpose="3e16 screen, repetition matched: expert threshold for up4 (2200 / 2600) and data "
    "cleaning (games under 10 plies, abandoned or unterminated games, provisional ratings)",
)

INPUTS = (dict(), dict(clock=True), dict(elo=True), dict(clock=True, elo=True))
AUX = ("aux_time", "aux_wdl")

WAVES["transfer"] = WAVES["wave2"] | dict(
    study="data-v1-transfer",
    prefix="dmixtr",
    runs=screen(
        runs("1e17", ["control"], (42, 43)) + runs("1e17", ["up4", "cooldown20"]),
        pool_frac=0.031,
    ),
    purpose="does the pf014 reversal of expert up-weighting hold at 1e17? repetition matched "
    "(pool_frac 0.031 = 1e17 tokens / 85B), paired with wave 2's full-pool 1e17 runs",
)

FINAL_TOKENS = 23e9  # next final run: 4B params x 23B tokens (user, 2026-09-18)


def pool_frac(budget):
    return SIZES[budget]["steps"] * 512 * 1024 / FINAL_TOKENS


WAVES["round2"] = WAVES["wave2"] | dict(
    study="data-v1-round2",
    prefix="r2",
    runs=screen(
        runs("3e16", ["control"], (42, 43, 44))
        + runs(
            "3e16",
            ["up2", "up4", "cooldown20", "mover_rule", "balanced_c10"]
            + ["clean_short", "clean_term", "provisional_down", "bots_masked"],
        )
        + [
            r | x
            for x in (
                dict(aux_time=0.2),
                dict(aux_wdl=0.2),
                dict(aux_time=0.2, aux_wdl=0.2),
            )
            for r in runs("3e16", ["control"])
        ]
        + [
            r | dict(clock=True) | x
            for x in (dict(), dict(input_lr=5.0))
            for r in runs("3e16", ["control"], (42, 43))
        ],
        pool_frac=pool_frac("3e16"),
        history=True,
    ),
    purpose="round 2 screen at 3e16 with the final run's repetition over all Lichess history "
    "(4B x 23B final run, 8.13B games): expert weighting (up2, up4, cooldown20), mover rule, cell "
    "balance, label cleaning, think-time / outcome aux heads at weight 0.2, and the clock input "
    "with its no-clock row fixed at zero (table lr ×75 and ×5)",
)

# finalized months when round-2 promotions were planned: data-v1 (minus the 2026-07 eval month) + 2024-08..12
R2P_MONTHS = [f"{STORES[0]}/{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)]
R2P_MONTHS = [
    d
    for d in R2P_MONTHS
    if d.rsplit("/", 1)[1] <= "2026-08" and not d.endswith("2026-07")
] + [f"{STORES[1]}/2024-{m:02d}" for m in range(8, 13)]

WAVES["round2p"] = WAVES["wave2"] | dict(
    study="data-v1-round2p",
    prefix="r2p",
    runs=screen(
        [
            r | dict(clock=True) | x
            for x in (
                dict(),
                dict(policy="cooldown20"),
                dict(policy="up4"),
                dict(aux_time=0.2, aux_wdl=0.2),
            )
            for r in runs("1e17", ["control"])
        ],
        pool_frac=pool_frac("1e17"),
        history=True,
        stores=STORES,
        months=R2P_MONTHS,
    ),
    purpose="round 2 promotions at 1e17 (4B x 23B all-history repetition, 24 pinned months): "
    "baseline = control + clock (fixed none row); up4 and cooldown20 promoted; aux heads cost check",
)

WAVES["round2p2"] = WAVES["round2p"] | dict(
    study="data-v1-round2p2",
    prefix="r2p",
    runs=screen(
        [r | dict(clock=True) for r in runs("1e17", ["mover_rule", "clean_term"])],
        pool_frac=pool_frac("1e17"),
        history=True,
        stores=STORES,
        months=R2P_MONTHS,
    ),
    purpose="round 2 promotions under the 1-sigma gate (user): mover_rule and clean_term at 1e17, "
    "same setup as round2p, judged against its control + clock",
)

WAVES["round2p3"] = WAVES["round2p2"] | dict(
    study="data-v1-round2p3",
    runs=screen(
        [r | dict(clock=True) for r in runs("1e17", ["balanced_c10"])],
        pool_frac=pool_frac("1e17"),
        history=True,
        stores=STORES,
        months=R2P_MONTHS,
    ),
    purpose="round 2 promotion: balanced_c10 at 1e17, same setup as round2p (+clock)",
)

CLOCKS = dict(bucket=dict(clock=True), feats=dict(feats=3, input_lr=5.0))


def stacks(weightings, extras, clocks):
    """Stack candidates at 1e17: each expert weighting x the confirmed extras x each clock
    variant, all with both aux heads (user: always on). mover_rule replaces the base keep rule."""
    head = ["mover_rule"] if "mover_rule" in extras else []
    tail = [e for e in extras if e != "mover_rule"]
    return [
        r | dict(policy="+".join([*head, w, *tail]), aux_time=0.2, aux_wdl=0.2) | c
        for w in weightings
        for c in clocks
        for r in runs("1e17", ["control"])
    ]


WAVES["round2p4"] = WAVES["round2p3"] | dict(
    study="data-v1-round2p4",
    runs=screen(
        [r | dict(clock=True) for r in runs("1e17", ["up2"])],
        pool_frac=pool_frac("1e17"),
        history=True,
        stores=STORES,
        months=R2P_MONTHS,
    ),
    purpose="round 2 promotion: up2 at 1e17, same setup as round2p (+clock)",
)

WAVES["round2s"] = WAVES["round2p4"] | dict(
    study="data-v1-round2s",
    prefix="r2s",
    # draft: weightings and extras are finalized from the 1e17 promotions before planning
    runs=screen(
        stacks(
            ["up4", "balanced_c10+up4"], ["mover_rule", "clean_term"], CLOCKS.values()
        )
        # matched baseline for the cf3 stacks (the bucket stacks use round2p's control+clk+aux)
        + [
            r | CLOCKS["feats"] | dict(aux_time=0.2, aux_wdl=0.2)
            for r in runs("1e17", ["control"])
        ],
        pool_frac=pool_frac("1e17"),
        history=True,
        stores=STORES,
        months=R2P_MONTHS,
    ),
    purpose="round 2 stacks at 1e17 (same pinned setup as the promotions): best expert "
    "weighting x confirmed extras x clock variant (bucket x75 / continuous cf3), both aux heads",
)

WAVES["scale1e16"] = WAVES["round2"] | dict(
    study="data-v1-scale1e16",
    prefix="s16",
    runs=screen(
        runs("1e16", ["control"], (42, 43))
        + runs(
            "1e16", ["up4", "cooldown20", "balanced_c10", "mover_rule", "clean_term"]
        )
        + [r | dict(feats=3, input_lr=5.0) for r in runs("1e16", ["control"])],
        pool_frac=pool_frac("1e16"),
        history=True,
    ),
    purpose="does 1e16 rank round-2 ideas like 3e16? same all-history 4B x 23B repetition and "
    "stores as round 2; if the ranking and gate decisions match, data screens move to 1e16",
)

WAVES["round2s2"] = WAVES["round2s"] | dict(
    study="data-v1-round2s2",
    # insurance: same-axis alternatives to up4, so B_2 does not hinge on up4 holding at 1e17
    runs=screen(
        stacks(
            ["up2", "cooldown20", "balanced_c10"],
            ["mover_rule", "clean_term"],
            [CLOCKS["bucket"]],
        ),
        pool_frac=pool_frac("1e17"),
        history=True,
        stores=STORES,
        months=R2P_MONTHS,
    ),
    purpose="round 2 insurance stacks at 1e17 on the general L40S lane: up2 / cooldown20 / "
    "balanced_c10 in place of up4, with mover_rule + clean_term, bucket clock, both aux heads",
)

WAVES["round2b"] = WAVES["round2"] | dict(
    study="data-v1-round2b",
    prefix="r2b",
    # judged against round 2's controls: no-feat batches are byte-identical (test_feats.py)
    runs=screen(
        [
            r | dict(feats=n, input_lr=5.0)
            for n in (1, 3)
            for r in runs("3e16", ["control"])
        ],
        pool_frac=pool_frac("3e16"),
        history=True,
    ),
    purpose="round 2 companion (same 4B x 23B all-history repetition): continuous clock "
    "features (log-Fourier of seconds) for the mover's time left, and with the opponent's "
    "time left and the mover's previous think time; feature table lr x5",
)
WAVES["inputs"] = WAVES["wave2"] | dict(
    study="data-v1-inputs",
    prefix="dmixin",
    runs=[r | x for x in INPUTS for r in runs("3e16", ["control"], (42, 43))],
    purpose="input ablations at 3e16 on data-v1 (2025-01..2026-08 minus 2026-07, full pool), paired "
    "with control: the mover's time left before each move, the mover's Elo bucket, and both",
)


# round 3 (2026-09-19): 3e16 screens on B_2 (data-v1-B2.json, baseline commit da0322b). B_2 x 3 seeds are
# the shared data / model-track controls (s42 general L40S, s43 / s44 preempt; dei is full), B_2 at 1e17
# is the kept-stack scale check (clean_term tripped the scale rule) and the model track's 1e17 control
B2_RUN = dict(policy="mover_rule+up4", feats=3, input_lr=5.0, aux_time=0.2, aux_wdl=0.2)
FAST = "RTX_PRO_6000|H100|H200|A100_80GB|A100_80G|L40S|6000Ada"
EXT = "/data/group_data/dei-group/yimingz3/allie/ext-v1"
EXT_MONTHS = [
    f"{EXT}/{m}"
    for m in (
        "otb/20xx-broadcast",
        "otb/20xx-pgnmentor",
        "otb/20xx-twic",
        "engine/20xx-ccrl404",
        "engine/20xx-ccrl4040",
        "engine/20xx-tcec",
    )
]


def b2(budget, policy=B2_RUN["policy"], seeds=(42,), **kw):
    """B_2 runs, optionally with another policy or other overrides."""
    return [
        r | B2_RUN | dict(policy=policy) | kw for r in runs(budget, [policy], seeds)
    ]


def r3(rs, budget="3e16", **kw):
    return screen(
        rs,
        pool_frac=pool_frac(budget),
        history=True,
        stores=kw.pop("stores", STORES),
        months=kw.pop("months", R2P_MONTHS),
        suffix=kw.pop("suffix", "b2"),
    )


WAVES["round3g"] = WAVES["wave2"] | dict(
    study="data-v1-round3g",
    prefix="r3g",
    b2=True,
    sbatch="""#SBATCH --account=dippolit
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-q9-32,babel-x9-32
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00""",
    runs=r3(b2("3e16")) + r3(b2("1e17"), "1e17"),
    purpose="round 3, general L40S lane: B_2 seed 42 at 3e16 (shared data / model control) and B_2 at "
    "1e17 (scale check of the kept stack, model-track 1e17 control)",
)

WAVES["round3p"] = WAVES["wave2"] | dict(
    study="data-v1-round3p",
    prefix="r3p",
    b2=True,
    throttle=6,
    sbatch=WAVES["wave2"]["sbatch"] + f"\n#SBATCH --constraint={FAST}",
    runs=r3(
        b2("3e16", seeds=(43, 44))
        + [
            r
            for p in (
                "mover_rule+up8",
                "mover_rule+up4_2200",
                "mover_rule+up4_cooldown20_up16",
                "mover_rule+balanced_c10+up4",
                "mover_rule+balanced_nobullet_c10+up4",
            )
            for r in b2("3e16", p)
        ]
        + b2("3e16", aux_time=0.1, aux_wdl=0.1)
        + b2("3e16", aux_time=0.05, aux_wdl=0.05)
    ),
    purpose="round 3 screen on B_2 (1 seed each, B_2 x 2 more seeds): expert weighting strength, "
    "threshold and cooldown (up8, up4 at 2200, up4 then up16 in the last 20%), cell balance on top "
    "of up4 with and without bullet, and cheaper aux heads (0.1, 0.05; aux cost 0.95x at 1e17)",
)

WAVES["round3x"] = WAVES["round3p"] | dict(
    study="data-v1-round3x",
    prefix="r3x",
    throttle=4,
    history_counts=ROOT / "results/recipe10x/data-v1-ext-history-counts.json",
    runs=r3(
        [
            r
            for p in (
                "mover_rule+up4",
                "mover_rule+up4+noengine",
                "mover_rule+up4+noengine+otb_x4",
                "mover_rule+up4+nootb",
                "mover_rule+up4+nootb+engine_x4",
                "mover_rule+up4+nootb+engine_cd_x10",
            )
            for r in b2("3e16", p)
        ],
        stores=[*STORES, f"{EXT}/otb", f"{EXT}/engine"],
        months=[*R2P_MONTHS, *EXT_MONTHS],
        suffix="b2x",
    ),
    purpose="round 3 external sources on B_2 (ext-v1: OTB broadcast / TWIC / PGN Mentor, engine "
    "TCEC / CCRL; merged all-history counts): both at natural weight, OTB only (x1, x4), engine only "
    "(x1, x4, x10 in the cooldown only); compared with the B_2 controls of round3g / round3p",
)


WAVES["round3c"] = WAVES["round3p"] | dict(
    study="data-v1-round3c",
    prefix="r3c",
    throttle=2,
    runs=r3(b2("3e16", seeds=(45, 46))),
    purpose="round 3 extra B_2 controls (s45, s46) at 3e16: B_2 seed spread is 0.0016 / 0.0024 golden "
    "macro / expert with 3 seeds (7-10x round 2's control); same sources as round 3, pooled by both tracks",
)


WAVES["round3x2"] = WAVES["round3x"] | dict(
    study="data-v1-round3x2",
    prefix="r3x2",
    throttle=3,
    runs=r3(
        [
            r
            for p in ("otb_x10", "otb_x30")
            for r in b2("3e16", f"mover_rule+up4+noengine+{p}")
        ]
        + b2("3e16", "mover_rule+up4+noengine+otb_x4", seeds=(43,)),
        stores=[*STORES, f"{EXT}/otb", f"{EXT}/engine"],
        months=[*R2P_MONTHS, *EXT_MONTHS],
        suffix="b2x",
    ),
    purpose="OTB dose-response on B_2: otb_x4 won (z -3.1 / -4.1); x10 and x30 find the peak, and x4 gets a "
    "second seed (s43); same sources and merged history as round3x",
)


WAVES["round3x3"] = WAVES["round3x2"] | dict(
    study="data-v1-round3x3",
    prefix="r3x3",
    throttle=1,
    runs=r3(
        b2("3e16", "mover_rule+up4+noengine+otb_x2"),
        stores=[*STORES, f"{EXT}/otb", f"{EXT}/engine"],
        months=[*R2P_MONTHS, *EXT_MONTHS],
        suffix="b2x",
    ),
    purpose="OTB dose below the peak: x1 null, x4 won, x10 and x30 lose (z +5.1 / +0.9, +36 / +26); x2 "
    "brackets the peak; same sources and merged history as round3x",
)

WAVES["round3s"] = WAVES["round3g"] | dict(
    study="data-v1-round3s",
    prefix="r3s",
    history_counts=ROOT / "results/recipe10x/data-v1-ext-history-counts.json",
    runs=r3(
        b2("1e17", "mover_rule+up4+noengine+otb_x4"),
        "1e17",
        stores=[*STORES, f"{EXT}/otb", f"{EXT}/engine"],
        months=[*R2P_MONTHS, *EXT_MONTHS],
        suffix="b2x",
    ),
    purpose="round 3 kept stack at 1e17: B_2 + OTB x4, the only 3e16 pass on the 5-seed B_2 (z -3.6 / -4.7); "
    "vs r3g's B_2 1e17; same sources and merged history as round3x",
)

WAVES["round3x4"] = WAVES["round3x2"] | dict(
    study="data-v1-round3x4",
    prefix="r3x4",
    throttle=2,
    runs=r3(
        b2("3e16", "mover_rule+up4+otb_x4+engine_x4")
        + b2("3e16", "mover_rule+up4+nootb+engine_x4", seeds=(43,)),
        stores=[*STORES, f"{EXT}/otb", f"{EXT}/engine"],
        months=[*R2P_MONTHS, *EXT_MONTHS],
        suffix="b2x",
    ),
    purpose="round 3 stack at 3e16: B_2 + OTB x4 + engine x4 (engine x4 passes narrowly, z +1.1 / -1.7; its "
    "leave-one-outs are the two single-source screens), and engine x4 seed 43",
)

B3C = "mover_rule+up4+noengine+otb_x4"
X = dict(
    stores=[*STORES, f"{EXT}/otb", f"{EXT}/engine"],
    months=[*R2P_MONTHS, *EXT_MONTHS],
    suffix="b3",
)
WAVES["round4"] = WAVES["round3x"] | dict(
    study="data-v1-round4",
    prefix="r4",
    b2=False,
    throttle=6,
    runs=r3(
        b2("3e16", B3C, seeds=(42, 43, 44, 45))
        + [
            r
            for p in (
                f"{B3C}+recent6_x2",
                f"{B3C}+recent6_x4",
                f"{B3C}+recent12_x2",
                "mover_rule+up4_cooldown20_up1+noengine+otb_x4",
            )
            for r in b2("3e16", p)
        ],
        **X,
    ),
    purpose="round 4 screen on the B_3 candidate B_2 + OTB x4 (its 1e17 check is round3s), 4 control seeds: "
    "recency within buckets (last 6 months x2 / x4, last 12 x2; golden is 2026-07) and up4 back to control "
    "in the last 20% (up16 there cost macro); sampler 211a8e6",
)

WAVES["round4s2"] = WAVES["round3g"] | dict(
    study="data-v1-round4s2",
    prefix="r4s2",
    sbatch=WAVES["wave2"]["sbatch"] + f"\n#SBATCH --constraint={FAST}",
    runs=r3(b2("1e17", seeds=(43,)), "1e17"),
    purpose="B_2 at 1e17 seed 43: second seed of the B_2 vs B_3 1e17 comparison (s42: 1.4272 / 1.3371 vs "
    "1.4251 / 1.3315); sampler 506e9fb, batches identical to da0322b's",
)

WAVES["round4s3"] = WAVES["round4"] | dict(
    study="data-v1-round4s3",
    prefix="r4s3",
    throttle=1,
    runs=r3(b2("1e17", B3C, seeds=(43,)), "1e17", **X),
    purpose="B_3 (B_2 + OTB x4) at 1e17 seed 43, paired with round4s2's B_2 s43",
)

WAVES["round5p"] = WAVES["round4"] | dict(
    study="data-v1-round5p",
    prefix="r5p",
    throttle=2,
    runs=r3(
        [
            r
            for p in (
                "mover_rule+up8+noengine+otb_x4",
                "mover_rule+up4+otb_x4+engine_x4",
            )
            for r in b2("1e17", p)
        ],
        "1e17",
        **X,
    ),
    purpose="round 5: borderline 3e16 expert-adding changes promoted to 1e17 on B_3 (OTB's 3e16 macro cost vanished "
    "at 1e17): up8 (5-seed z +2.2 / -2.8) and engine x4 (1-seed pass, 2-seed macro fail); vs B_3 s42 / s43",
)

def name(w, r):
    tag = f"-{r['tag']}" if "tag" in r else ""
    clk = (
        ("-clk" if r.get("clock") else "")
        + ("-elo" if r.get("elo") else "")
        + (f"-cf{r['feats']}" if r.get("feats") else "")
        + (f"-lr{r['input_lr']:g}" if "input_lr" in r else "")
        + "".join(
            f"-{k[4]}{round(100 * r[k]):02d}" for k in AUX if r.get(k)
        )  # -t20 = think-time head at weight 0.2
    )
    policy = r["policy"].replace(
        "+", "__"
    )  # composed policies; run names allow [A-Za-z0-9_-]
    return f"{w['prefix']}-{r['budget']}-{policy}{clk}{tag}-s{r['seed']}"


def plan(wave):
    w = WAVES[wave]
    study = ROOT / "results/recipe10x" / w["study"]
    assert not study.exists(), "never overwrite a frozen study"
    for d in ("source-ours", "evaluator-ours", "logs", "results"):
        (study / d).mkdir(parents=True)
    # this checkout's code (a worktree); results stay in ROOT
    src = Path(__file__).resolve().parent
    for f in [
        *src.glob("modded_*.py"),
        *(
            src / x
            for x in ("lm_data.py", "lm_checkpoint.py", "chessmix.py", "chess_vocab.py")
        ),
    ]:
        shutil.copy2(f, study / "source-ours")
    for f in ("eval_modded.py", "ce_alignment.py", "modded_runtime.py"):
        shutil.copy2(OLD_EVAL / f, study / "evaluator-ours")
    shutil.copy2(src / "eval_strat.py", study / "evaluator-ours")
    shutil.copy2(__file__, study / "dataexp.py")
    if any(r.get("history") for r in w["runs"]):
        shutil.copy2(w.get("history_counts", HISTORY), study / "history-counts.json")
    (study / "plan.json").write_text(
        json.dumps(
            dict(
                wave=wave,
                purpose=w["purpose"],
                sizes=SIZES,
                schedule=SCHEDULE,
                runs=[r | dict(name=name(w, r)) for r in w["runs"]],
                selection="golden strat-eval-v1 macro / expert macro",
                **b2_pin(w),
            ),
            indent=2,
        )
        + "\n"
    )
    (study / "run.sbatch").write_text(f"""#!/bin/bash
#SBATCH --job-name={w["prefix"]}
{w["sbatch"]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{-(-len(w["runs"]) // w.get("pack", 1)) - 1}{"%" + str(w["throttle"]) if w.get("throttle") else ""}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={study}/logs/%x-%A_%a.out
exec {ROOT}/.venv/bin/python {study}/dataexp.py task {wave}
""")


def submit(wave):
    study = ROOT / "results/recipe10x" / WAVES[wave]["study"]
    assert not (study / "submitted.json").exists(), "already submitted"
    job = subprocess.check_output(
        ["sbatch", "--parsable", str(study / "run.sbatch")], text=True
    ).strip()
    (study / "submitted.json").write_text(
        json.dumps(dict(at=time.time(), job=job)) + "\n"
    )
    print(job)


def task_id():
    return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"


def seconds_left():
    left = subprocess.check_output(
        ["squeue", "-h", "-o", "%L", "-j", task_id()], text=True
    ).strip()
    days, _, hms = left.rpartition("-")
    seconds = 0
    for x in hms.split(":"):
        seconds = seconds * 60 + int(x)
    return seconds + 86400 * int(days or 0)


def run(cmd, env, log):
    with open(log, "a") as f:
        subprocess.run(
            [str(x) for x in cmd],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def task(wave):
    """Run this array task's slice of `pack` runs concurrently, spread over its `gpus` GPUs."""
    w = WAVES[wave]
    study = ROOT / "results/recipe10x" / w["study"]
    k, gpus, rg = w.get("pack", 1), w.get("gpus", 1), w.get("run_gpus", 1)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    mine = w["runs"][i * k : (i + 1) * k]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    slots = [
        ",".join(visible[s : s + rg]) for s in range(0, gpus, rg)
    ]  # rg GPUs per run
    with ThreadPoolExecutor(len(mine)) as ex:
        finished = list(
            ex.map(
                lambda j: run_one(w, study, mine[j], slots[j % len(slots)]),
                range(len(mine)),
            )
        )
    if not all(finished):
        subprocess.run(["scontrol", "requeue", task_id()], check=True)


def run_one(w, study, r, gpu):
    """Train, then score one run. False when it stopped early and the task must be requeued."""
    size, n = SIZES[r["budget"]], name(w, r)
    result = study / "results" / f"{n}.json"
    if result.exists():
        return True
    started, left = time.monotonic(), seconds_left()
    env = (
        os.environ
        | dict(
            ALLIE_PROJECT_ROOT=str(ROOT),
            CUDA_VISIBLE_DEVICES=gpu,
            OMP_NUM_THREADS="4",
            PYTHONUNBUFFERED="1",
            TORCHINDUCTOR_COMPILE_THREADS="4",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            CUDA_MODULE_LOADING="LAZY",
            DISABLE_FP8="1",
            NCCL_CUMEM_HOST_ENABLE="0",
            NCCL_IB_DISABLE="1",
            NCCL_P2P_DISABLE="1",
            TORCHINDUCTOR_CACHE_DIR=f"/scratch/yimingz3/allie/{w['prefix']}-inductor",
            TRITON_CACHE_DIR=f"/scratch/yimingz3/allie/{w['prefix']}-triton",
            PYTHONPATH="/data/group_data/dei-group/yimingz3/allie/envs/chessmix-overlay",  # pyarrow for the pinned runtime
        )
    )
    src = study / "source-ours"
    py = subprocess.check_output(
        [sys.executable, src / "modded_runtime_stage.py"], env=env, text=True
    ).strip()
    out = ROOT / "results/pretrain" / n
    done = out / "done.json"
    log = study / "logs" / n
    if not (done.exists() and json.loads(done.read_text())["stop_reason"] == "steps"):
        cmd = [
            py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpu.split(','))}", src / "modded_train.py",
            "--name", n,
            "--width", size["width"],
            "--layers", size["layers"],
            "--head-dim", 64,
            "--steps", size["steps"],
            "--extension-steps", 0,
            "--initial-batch-rows", 512,
            "--micro-batch", w.get("micro_batch", 16),
            "--lr-scale", 1,
            "--seed", r["seed"],
            "--eval-every", 10**7,
            "--checkpoint-every", 128,  # halves work lost to preemption (barrier)
            "--keep-checkpoints", 2,
            "--val-rows", 1024,
            "--max-seconds", max(600, left - int(time.monotonic() - started) - 1500),
            "--deterministic",
            "--data", DATA,
            "--mix", r["policy"],
            "--wsd-schedule", json.dumps(SCHEDULE, sort_keys=True),
            "--wsd-end-step", size["steps"],
            "--wsd-decay-start", 32,
        ]  # fmt: skip
        if "pool_frac" in r:
            cmd += ["--mix-pool-frac", r["pool_frac"]]
        if "stores" in r:
            cmd += ["--mix-stores", ",".join(r["stores"])]
        if r.get("history"):
            cmd += ["--mix-history", study / "history-counts.json"]
        if r.get("months"):
            cmd += ["--mix-months", ",".join(r["months"])]
        if r.get("clock"):
            cmd += ["--clock"]
        if r.get("elo"):
            cmd += ["--elo"]
        if r.get("feats"):
            cmd += ["--clock-feats", r["feats"]]
        if "input_lr" in r:
            cmd += ["--input-lr-mul", r["input_lr"]]
        for k in AUX:
            if r.get(k):
                cmd += [f"--{k.replace('_', '-')}", r[k]]
        if (out / "last.pt").exists():
            cmd += ["--resume", out / "last.pt"]
        run(cmd, env, f"{log}.train.log")
        if json.loads(done.read_text())["stop_reason"] != "steps":
            return False
    report = ROOT / "results/lm-eval" / n / "original-val.json"
    if not report.exists():
        run(
            [
                py,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
                study
                / "evaluator-ours/eval_strat.py",  # superset of eval_modded; feeds clock/Elo inputs
                "--checkpoint",
                out / "last.pt",
                "--source",
                src,
                "--split",
                "original_val",
                "--batch",
                16,
            ],  # fmt: skip
            env,
            f"{log}.eval.log",
        )
    ev = json.loads(report.read_text())
    assert [ev[k + "_count"] for k in ("move", "expert2400", "expert2600")] == [
        4616637,
        396483,
        151660,
    ]
    strat = ROOT / "results/lm-eval" / n / "strat-v1.json"
    if not strat.exists():
        run(
            [
                py,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
                study / "evaluator-ours/eval_strat.py",
                "--checkpoint",
                out / "last.pt",
                "--source",
                src,
                "--split",
                "strat",
                "--batch",
                16,
            ],  # fmt: skip
            env,
            f"{log}.strat.log",
        )
    sv = json.loads(strat.read_text())
    pin = json.loads((study / "plan.json").read_text())
    tmp = result.with_suffix(".tmp")  # atomic: a preempted write never looks complete
    tmp.write_text(
        json.dumps(
            r
            | {k: pin[k] for k in ("b2_sha256",) if k in pin}
            | dict(
                name=n,
                ce={k: ev[k + "_ce"] for k in ("move", "expert2400", "expert2600")},
                strat=dict(
                    macro=sv["macro"],
                    expert_macro=sv["expert_macro"],
                    cells=sv["cells"],
                ),
                report=str(report),
                job=os.environ["SLURM_JOB_ID"],
                node=os.uname().nodename,
                task_seconds=time.monotonic() - started,
            ),
            indent=2,
        )
        + "\n"
    )
    tmp.replace(result)
    return True


def status(wave):
    w = WAVES[wave]
    study = ROOT / "results/recipe10x" / w["study"]
    for r in w["runs"]:
        n, (bm, be), steps = (
            name(w, r),
            SIZES[r["budget"]]["base"],
            SIZES[r["budget"]]["steps"],
        )
        res, log = study / "results" / f"{n}.json", study / "logs" / f"{n}.train.log"
        if res.exists():
            ce = json.loads(res.read_text())["ce"]
            print(
                f"{n:34} move {ce['move']:.4f} ({ce['move'] - bm:+.4f})  >=2400 {ce['expert2400']:.4f} ({ce['expert2400'] - be:+.4f})"
            )
            continue
        seen = (
            [
                json.loads(l)["step"]
                for l in open(log)
                if l.startswith('{"step"') and "train_ce" in l
            ]
            if log.exists()
            else []
        )
        print(f"{n:34} step {seen[-1] if seen else 0}/{steps}")


if __name__ == "__main__":
    dict(plan=plan, submit=submit, task=task, status=status)[sys.argv[1]](sys.argv[2])
