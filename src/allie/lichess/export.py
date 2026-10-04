"""Training checkpoint (last.pt pointer or model.pt) -> model.safetensors + config.json.

Matrices go to F.linear layout with the attention lambdas folded in, rounded to BF16 as the
training forward rounds them; router, balancing bias, router centre and scalars stay FP32.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.torch_version import TorchVersion

from .tokens import CONTEXT


def load(checkpoint):
    """The checkpoint state, following a last.pt / best.pt pointer, and the model file's path."""
    path = Path(checkpoint)
    with torch.serialization.safe_globals([TorchVersion]):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if "directory" in state:
            path = path.parent / state["directory"] / "model.pt"
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return state, path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def convert(state):
    cfg, m, inf = state["config"], state["model"], state["inference"]
    arch = cfg["arch"]
    n, d = cfg["layers"], cfg["width"]
    assert cfg["feats"] == 3 and inf["split_embed"]
    # the switches this port implements; anything else would be exported as something else
    assert arch.get("board") == "conv" and arch.get("mlp") == "swiglu"
    assert arch.get("tc_header", True) and arch.get("x0", True)
    assert arch.get("moe_score", "sigmoid") == "sigmoid" and arch.get("moe_shared", True)
    for key in ("key_offset", "header_feats", "moe_log_gates", "moe_keep"):
        assert not arch.get(key), f"arch {key} is not supported"
    assert not arch.get("moe_router_center_first"), "arch moe_router_center_first"
    assert min(inf["ws_short"], inf["ws_long"]) * 128 >= CONTEXT  # plain causal per game
    s = m["scalars"].float()
    bf = lambda t: t.bfloat16().contiguous()
    w = {
        "embed": m["embed.weight"],
        "embed2": m["embed2.weight"],
        "lm_head": m["lm_head.weight"],
        "feat_embed": m["feat_embed.weight"],
        "smear_gate": m["smear_gate.weight"],
        "scalars": s,
        "x0_lambdas": m["x0_lambdas"].float(),
        "cos": inf["yarn"]["cos"][:CONTEXT],
        "sin": inf["yarn"]["sin"][:CONTEXT],
    }
    w |= {f"skip_gate.{j}": m[f"skip_gates.{j}.weight"] for j in range(3)}
    ve = sum(k.startswith("value_embeds.") for k in m)
    w |= {f"value_embed.{j}": m[f"value_embeds.{j}.weight"] for j in range(ve)}
    w |= {f"board.{k[6:]}": v for k, v in m.items() if k.startswith("board.")}
    for i in range(n):
        b = f"blocks.{i}."
        qkvo = m[b + "attn.qkvo_w"].float()
        w[f"{i}.qkv"] = qkvo[: 3 * d] * s[n + 2 * i]
        w[f"{i}.o"] = qkvo[3 * d :] * s[n + 2 * i + 1]
        w[f"{i}.attn_gate"] = m[b + "attn.attn_gate.weight"]
        if b + "attn.value_embed_gate.weight" in m:
            w[f"{i}.ve_gate"] = m[b + "attn.value_embed_gate.weight"]
        if b + "mlp.c_fc" in m:
            w[f"{i}.fc"], w[f"{i}.proj"] = m[b + "mlp.c_fc"], m[b + "mlp.c_proj"].T
            continue
        w[f"{i}.router"] = m[b + "mlp.router"].float()
        w[f"{i}.moe_bias"] = m[b + "mlp.bias"].float()
        w[f"{i}.mu"] = m.get(b + "mlp.mu", torch.zeros(d)).float()  # router centre
        w[f"{i}.up"] = m[b + "mlp.up"]
        w[f"{i}.down"] = m[b + "mlp.down"].transpose(1, 2)
        w[f"{i}.shared_up"] = m[b + "mlp.shared_up"]
        w[f"{i}.shared_down"] = m[b + "mlp.shared_down"].T
    fp32 = ("router", "moe_bias", "mu", "scalars", "x0_lambdas")
    w = {k: v.contiguous() if k.split(".")[-1] in fp32 else bf(v) for k, v in w.items()}
    experts, topk = arch["moe"]
    up = w["1.up"]
    config = dict(
        layers=n,
        width=d,
        heads=d // cfg["head_dim"],
        head_dim=cfg["head_dim"],
        vocab=w["lm_head"].shape[0],
        value_embeds=ve,
        experts=experts,
        topk=topk,
        expert_hidden=up.shape[1] // 2,
        shared_hidden=w["1.shared_up"].shape[0] // 2,
        gate_floor=arch.get("moe_gate_floor", 0.0),
        attn_scale=inf["yarn"]["attn_scale"],
    )
    assert up.shape[0] == experts and config["heads"] * cfg["head_dim"] == d
    return w, config


def main(checkpoint, out, name="allie-2.0"):
    state, path = load(checkpoint)
    w, config = convert(state)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(w, str(out / "model.safetensors"))
    config |= dict(
        name=name,
        step=state["step"],
        checkpoint=str(path.resolve()),
        checkpoint_sha256=sha256(path),
    )
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    return config
