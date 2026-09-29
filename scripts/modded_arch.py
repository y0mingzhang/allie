"""Model-track architecture switches carried in Config.arch (torch-free: modelexp imports it too).

The recipe is the big run's: SwiGLU MLPs, the board CNN input, QK norm, gates, x0 and a second
embedding, softcapped logits, NorMuon with cautious weight decay. What stays switchable is the MoE.
"""

DEFAULTS = dict(
    moe=None,  # [experts, top-k]: DeepSeek MoE MLPs after the first layer (modded_moe)
    # router init std, router Adam lr multiplier, bias update speed, sequence-wise balance loss weight
    moe_init=0.006,
    moe_router_lr_mul=0.1,
    moe_router_wd=0.0,  # decoupled (AdamW) router weight decay: w -= lr * wd * w per Adam step
    moe_gamma=1e-2,
    moe_seq=1e-3,
    moe_seq_raw=False,  # balance loss counts from the raw top-k (DeepSeek-V3 Eq. 18), not the biased one
    moe_router_center=0.0,  # EMA decay of the mean router input subtracted before the router (0: off)
    moe_update="prop",  # prop | sign | quantile (Kimi K3's Quantile Balancing: moe_gamma unused)
    moe_score="sigmoid",  # sigmoid (DeepSeek-V3) | sqrtsoftplus (DeepSeek-V4.1 Flash)
    moe_shared=True,  # shared expert; off: routed experts take the whole active width
    moe_shared_frac=0.5,  # the shared expert's share of the active width (1 / (k + 1) = DeepSeek's uniform)
    moe_round=0,  # round shared and routed widths to multiples of this (0: exact split)
    # FP32 masters and update math for the BF16 weights that have none: head, embeddings, gates
    fp32_small_masters=False,
    x0=True,  # off: drop the x0 re-injection, its blend weights held at 0 (screen-1 nox0)
    # per-token header features (modded_medium_core.header_features) on every move position:
    # 1 = mover's and opponent's Elo, 2 = + the base time and increment
    header_feats=0,
    header_lr_mul=None,  # Adam lr multiplier of the header table (None: input_lr_mul)
    # Adam and scalar optimizers step every step at half their lr, twice their (lr^2) weight decay
    # and square-rooted betas (off: odd steps only, on two steps of summed gradient, as upstream)
    adam_every=False,
    # the header's base-time and increment tokens (False: every forward replaces them with the
    # unknown-time-control tokens, so the model never sees them; the clock features stay)
    tc_header=True,
    wd_scale=1.0,  # multiplies every optimizer group's (lr^2, cautious) weight decay; not moe_router_wd
    # floor of the sums of sigmoid scores that renormalise a token's gates and its balance-loss affinities
    # (0: off); modded_moe.MoE gate_floor
    moe_gate_floor=0.0,
)
# switches a --resume-new-source resume may change: numerical guards, not a different model
RESUMABLE = ("moe_gate_floor",)
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
    moe_shard=False,
    diff_attn=False,
    aux_detach=False,
)


def resolve(arch):
    unknown = set(arch) - set(DEFAULTS) - set(SHIPPED)
    assert not unknown, f"unknown arch keys {sorted(unknown)}"
    retired = {k: v for k, v in arch.items() if k in SHIPPED and v != SHIPPED[k]}
    shipped = {k: SHIPPED[k] for k in retired}
    assert not retired, (
        f"retired arch switches {retired}: this code builds only {shipped}"
    )
    out = DEFAULTS | {k: v for k, v in arch.items() if k in DEFAULTS}
    assert out["moe_update"] in ("sign", "prop", "quantile")
    assert out["moe_score"] in ("sigmoid", "sqrtsoftplus")
    assert out["header_feats"] in (0, 1, 2)
    assert out["wd_scale"] >= 0 and isinstance(out["adam_every"], bool)
    assert out["tc_header"] or out["header_feats"] < 2, (
        "header_feats 2 reads the masked tokens"
    )
    assert 0 <= out["moe_gate_floor"] < float("inf")
    return out


def moe_dims(width, arch):
    """MoE constructor arguments: the dense MLP's active hidden width (swiglu_hidden) split into a
    shared expert of half of it plus k routed experts sharing the rest (all routed without the
    shared expert), then the routing settings."""
    a = resolve(arch)
    if not a["moe"]:
        return None
    experts, topk = a["moe"]
    active = swiglu_hidden(width)
    frac, m = (a["moe_shared_frac"] if a["moe_shared"] else 0), a["moe_round"] or 1
    shared = active // 2 if frac == 0.5 and m == 1 else round(active * frac / m) * m
    return (
        experts,
        topk,
        round((active - shared) / topk / m) * m,  # extra_flops counts the rounding
        shared,
        *(a[k] for k in ("moe_init", "moe_router_lr_mul", "moe_gamma", "moe_seq")),
        a["moe_update"],
        a["moe_score"],
        a["moe_seq_raw"],
        a["moe_router_wd"],
        a["moe_router_center"],
        a["moe_gate_floor"],
    )


def swiglu_hidden(width):
    """Hidden width giving SwiGLU (3 matrices) the relu2 MLP's 8 d^2 parameters."""
    return round(8 * width / 3 / 16) * 16


def board_macs(width):
    """Dense multiply-adds per position of the board CNN (Codex's modded_spatial)."""
    return 64 * (13 * 32 * 9 + 2 * 32 * 32 * 9 + 32 * 8) + 32 * 32 + 544 * width


def extra_flops(arch, width, layers):
    """Forward FLOPs per token beyond modded_train.useful_flops' dense relu2 count (may be negative):
    the board CNN, SwiGLU MLPs and the MoE (nominal: k experts per token, plus the routers)."""
    a = resolve(arch)
    extra, dense = 2 * board_macs(width), layers
    if a["moe"]:
        _, k, routed, shared, *_ = moe_dims(width, arch)
        extra += 2 * (layers - 1) * width * a["moe"][0]
        extra += (
            2 * (layers - 1) * (3 * width * (shared + k * routed) - 8 * width * width)
        )
        dense = 1
    return extra + 2 * dense * (3 * width * swiglu_hidden(width) - 8 * width * width)
