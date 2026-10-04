"""Architecture switches carried in Config.arch (torch-free: experiments.modelexp imports it).

The fixed recipe is Allie 2.0's: SwiGLU MLPs, the board CNN input, QK norm, gates, x0 and a second
embedding, softcapped logits, NorMuon with cautious weight decay, MoE layers after dense blocks
balanced by Quantile Balancing with a shared expert. What stays switchable is the MoE's shape and routing.
"""

DEFAULTS = dict(
    moe=None,  # [experts, top-k]: DeepSeek MoE MLPs after moe_dense_first dense blocks (model.moe)
    # router init std, router Adam lr multiplier, sequence-wise balance loss weight
    moe_init=0.006,
    moe_router_lr_mul=0.1,
    moe_seq=1e-3,
    moe_router_center=0.0,  # EMA decay of the mean router input subtracted before the router (0: off)
    moe_router_center_first=0,  # centre only the first n MoE layers' router inputs (0: every MoE layer)
    moe_shared_frac=0.5,  # the shared expert's share of the active width (1 / (k + 1) = DeepSeek's uniform)
    moe_round=0,  # round shared and routed widths to multiples of this (0: exact split)
    # FP32 masters and update math for the BF16 weights that have none: head, embeddings, gates
    fp32_small_masters=False,
    # Adam and scalar optimizers step every step at half their lr, twice their (lr^2) weight decay
    # and square-rooted betas (off: odd steps only, on two steps of summed gradient, as upstream)
    adam_every=False,
    # floor of the sums of sigmoid scores that renormalise a token's gates and its balance-loss affinities
    # (0: off); model.moe.MoE gate_floor
    moe_gate_floor=0.0,
    # gates as k**0.5 * softmax of the selected log-sigmoid scores (and balance-loss affinities as the softmax of all of
    # them): the same function as the renormalised sigmoids, finite in forward and backward at any score
    moe_log_gates=False,
    moe_dense_first=1,  # blocks 0 .. n-1 keep the dense MLP, MoE after (DeepSeek-V3: 3)
)
# switches a --resume-new-source resume may change: numerical guards, not a different model
RESUMABLE = ("moe_gate_floor", "moe_log_gates")
# Retired switches, accepted only at the value this code hardcodes: older configs that set anything
# else describe a different model.
SHIPPED = dict(
    qk_norm=True,
    gates=True,
    key_offset=False,
    full_rope=False,
    mlp="swiglu",
    embed2=True,
    softcap=True,
    plain_init=False,
    untie_ve=False,
    matrix_adam=0.0,
    matrix_wd=0.005,
    uniform_mults=False,
    fp32_embed=False,
    cautious_wd=True,
    normuon=True,
    board="conv",
    moe_capacity=1.25,
    moe_kernel="scatter-dualgather",
    diff_attn=False,
    aux_detach=False,
    moe_update="quantile",
    moe_gamma=1e-2,
    moe_score="sigmoid",
    moe_seq_raw=False,
    moe_router_wd=0.0,
    moe_shared=True,
    x0=True,
    header_feats=0,
    header_lr_mul=None,
    tc_header=True,
    wd_scale=1.0,
    moe_keep=0,
    moe_shard=False,
)


def resolve(arch):
    unknown = set(arch) - set(DEFAULTS) - set(SHIPPED)
    assert not unknown, f"unknown arch keys {sorted(unknown)}"
    retired = {k: v for k, v in arch.items() if k in SHIPPED and v != SHIPPED[k]}
    shipped = {k: SHIPPED[k] for k in retired}
    assert not retired, (
        f"retired arch switches {retired}: this code builds only {shipped}"
    )
    # an MoE config without moe_update balanced by the retired default rule (prop)
    assert not arch.get("moe") or "moe_update" in arch, "MoE configs set moe_update"
    out = DEFAULTS | {k: v for k, v in arch.items() if k in DEFAULTS}
    assert isinstance(out["adam_every"], bool)
    assert 0 <= out["moe_gate_floor"] < float("inf")
    assert isinstance(out["moe_log_gates"], bool)
    assert isinstance(out["moe_dense_first"], int) and out["moe_dense_first"] >= 1
    assert isinstance(out["moe_router_center_first"], int) and out["moe_router_center_first"] >= 0
    return out


def moe_dims(width, arch, layer=0):
    """MoE constructor arguments of MoE layer `layer` (0 = the first): the dense MLP's active hidden
    width (swiglu_hidden) split into a shared expert of moe_shared_frac of it plus k routed experts
    sharing the rest, then the routing settings."""
    a = resolve(arch)
    first = a["moe_router_center_first"]
    if not a["moe"]:
        return None
    experts, topk = a["moe"]
    active = swiglu_hidden(width)
    frac, m = a["moe_shared_frac"], a["moe_round"] or 1
    shared = active // 2 if frac == 0.5 and m == 1 else round(active * frac / m) * m
    return (
        experts,
        topk,
        round((active - shared) / topk / m) * m,  # extra_flops counts the rounding
        shared,
        *(a[k] for k in ("moe_init", "moe_router_lr_mul", "moe_seq")),
        a["moe_router_center"] if not first or layer < first else 0.0,
        a["moe_gate_floor"],
        a["moe_log_gates"],
    )


def swiglu_hidden(width):
    """Hidden width giving SwiGLU (3 matrices) the relu2 MLP's 8 d^2 parameters."""
    return round(8 * width / 3 / 16) * 16


def board_macs(width):
    """Dense multiply-adds per position of the board CNN (model.board)."""
    return 64 * (13 * 32 * 9 + 2 * 32 * 32 * 9 + 32 * 8) + 32 * 32 + 544 * width


def extra_flops(arch, width, layers):
    """Forward FLOPs per token beyond train.trainer.useful_flops' dense relu2 count (may be negative):
    the board CNN, SwiGLU MLPs and the MoE (nominal: k experts per token, plus the routers)."""
    a = resolve(arch)
    extra, dense = 2 * board_macs(width), layers
    if a["moe"]:
        _, k, routed, shared, *_ = moe_dims(width, arch)
        dense = a["moe_dense_first"]
        assert dense <= layers, "moe_dense_first exceeds the layers"
        m = layers - dense
        extra += 2 * m * width * a["moe"][0]
        extra += 2 * m * (3 * width * (shared + k * routed) - 8 * width * width)
    return extra + 2 * dense * (3 * width * swiglu_hidden(width) - 8 * width * width)
