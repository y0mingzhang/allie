"""Model-track architecture switches carried in Config.arch (torch-free: modelexp imports it too).

An empty arch is the inherited modded-nanogpt recipe; every key defaults to that behaviour.
"""

DEFAULTS = dict(
    qk_norm=True,  # RMS-norm q and k
    gates=True,  # attention output gate and value-embedding gate
    key_offset=True,  # 1-layer-induction key shift on long-window layers
    full_rope=False,  # rotate every head dim (default: half-truncated RoPE)
    mlp="relu2",  # relu2 | gelu | swiglu
    x0=True,  # re-inject the input embedding x0 at every layer
    embed2=True,  # second input embedding x02 mixed in at every layer
    softcap=True,  # logits 23 * sigmoid((z + 5) / 7.5)
    plain_init=False,  # residual lambdas 1.0 and attention scale 1/sqrt(head_dim)
    untie_ve=False,  # one value-embedding table per layer instead of layers i and i+k sharing
    matrix_adam=0.0,  # > 0: AdamW for attention/MLP matrices instead of NorMuon, at this base lr
    # (the WSD schedule multiplies base lrs by up to its plateau, 4.0 in the model track)
    matrix_wd=0.005,  # their DistAdam weight decay: each step decays by lr^2 * wd
    adam_every=False,  # Adam groups step every step (default: odd steps only)
    uniform_mults=False,  # embed2 lr/wd multipliers and embed/lm_head wd multiplier -> 1
    fp32_embed=False,  # FP32 embedding/head params and Adam state, BF16 forward
    cautious_wd=True,  # cautious (sign-gated) decoupled weight decay
    normuon=True,  # NorMuon second-moment variance reduction on the Muon update
    board=None,  # None | direct | conv: board-state input at every position (modded_board)
    moe=None,  # [experts, top-k]: DeepSeek MoE MLPs after the first layer (modded_moe)
    # MoE routing: router init std, router Adam lr multiplier, bias update speed, sequence-wise balance
    # loss weight (pilot2's defaults collapsed; tier 2b tunes them)
    moe_init=0.02,
    moe_router_lr_mul=0.1,
    moe_gamma=1e-3,
    moe_seq=0.0,
    moe_update="sign",
    moe_capacity=1.25,  # training capacity factor (eval is dropless)
    moe_shared=True,  # shared expert (half the active width); off: routed experts take all of it
    moe_shared_frac=0.5,  # the shared expert's share of the active width (1 / (k + 1) = DeepSeek's uniform)
    moe_round=0,  # round shared and routed widths to multiples of this (0: exact split)
    moe_score="sigmoid",  # sigmoid (DeepSeek-V3) | sqrtsoftplus (DeepSeek-V4.1 Flash)
    moe_kernel="pad",  # pad (capacity bmm) | scatter (ScatterMoE, dropless)
    moe_shard=False,  # experts sharded over ranks (whole experts), gathered per layer
    diff_attn=False,  # differential attention (modded_diffattn)
    aux_detach=False,  # think-time / W-D-L head rows read stop-gradient features (trunk unaffected)
)


def resolve(arch):
    unknown = set(arch) - set(DEFAULTS)
    assert not unknown, f"unknown arch keys {unknown}"
    out = DEFAULTS | arch
    assert out["mlp"] in ("relu2", "gelu", "swiglu") and out["board"] in (
        None,
        "direct",
        "conv",
    )
    # the key offset shifts exactly the dims half-truncated RoPE leaves stationary
    assert not (out["full_rope"] and out["key_offset"]), (
        "full_rope needs key_offset=False"
    )
    assert not out["moe"] or (out["mlp"] in ("relu2", "swiglu") and not out["matrix_adam"])
    return out


def moe_dims(width, arch):
    """MoE constructor arguments: the dense MLP's active hidden width (4d, or swiglu_hidden) split
    into a shared expert of half of it plus k routed experts sharing the rest (all routed without
    the shared expert), then the routing settings, training capacity and expert kind."""
    a = resolve(arch)
    if not a["moe"]:
        return None
    experts, topk = a["moe"]
    active = swiglu_hidden(width) if a["mlp"] == "swiglu" else 4 * width
    frac, m = (a["moe_shared_frac"] if a["moe_shared"] else 0), a["moe_round"] or 1
    shared = active // 2 if frac == 0.5 and m == 1 else round(active * frac / m) * m
    return (
        experts,
        topk,
        round((active - shared) / topk / m) * m,  # extra_flops counts the rounding
        shared,
        *(a[k] for k in ("moe_init", "moe_router_lr_mul", "moe_gamma", "moe_seq")),
        a["moe_update"],
        a["moe_capacity"],
        a["mlp"],
        a["moe_score"],
        a["moe_kernel"],
        a["moe_shard"],
    )


def attn_factor(arch):
    """Attention score / value FLOPs relative to standard attention."""
    return 2 if resolve(arch)["diff_attn"] else 1


def swiglu_hidden(width):
    """Hidden width giving SwiGLU (3 matrices) the relu2 MLP's 8 d^2 parameters."""
    return round(8 * width / 3 / 16) * 16


AUX_ROWS = (
    2432 - 2350
)  # padded head rows from the first think-time bin (modded_medium.VOCAB)
BOARD_FEATURES = (
    64 * 13 + 2 + 16 + 9
)  # pieces one-hot, side, castling rights, en-passant file
BOARD_IN = BOARD_FEATURES + 5  # padded to a multiple of 8


def board_macs(kind, width):
    """Dense multiply-adds per position of the board branch (Codex's modded_spatial for conv)."""
    if kind == "direct":
        return BOARD_IN * width
    if kind == "conv":
        return 64 * (13 * 32 * 9 + 2 * 32 * 32 * 9 + 32 * 8) + 32 * 32 + 544 * width
    return 0


def extra_flops(arch, width, layers):
    """Forward FLOPs per token beyond modded_train.useful_flops' dense count (may be negative).
    The board's one-hot input matmul is counted like any matmul (x3 for training), though it needs
    no input gradient: +0.5% FLOPs for boarddirect, against the board arms."""
    a = resolve(arch)
    extra = 2 * board_macs(a["board"], width)
    if a[
        "aux_detach"
    ]:  # the aux head rows' second (detached) matmul: forward, x3 like every
        extra += (
            2 * width * AUX_ROWS
        )  # matmul, though it has no input gradient (+0.1% at 8 x 512)
    dense = layers
    if a["moe"]:  # nominal: k experts per token; dropped routes (logged) do no MLP work
        _, k, routed, shared, *_ = moe_dims(width, arch)
        per = 3 if a["mlp"] == "swiglu" else 2
        extra += 2 * (layers - 1) * width * a["moe"][0]  # routers
        extra += 2 * (layers - 1) * (per * width * (shared + k * routed) - 8 * width * width)
        dense = 1
    if a["mlp"] == "swiglu":
        extra += 2 * dense * (3 * width * swiglu_hidden(width) - 8 * width * width)
    return extra
