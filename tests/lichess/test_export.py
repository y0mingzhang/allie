import json

import pytest
import torch
from safetensors.torch import load_file

from allie.lichess.export import convert

from .conftest import C4X, tiny_export

ALLIE20 = dict(adam_every=True, board="conv", fp32_small_masters=True, key_offset=False, mlp="swiglu", moe=[256, 16],
               moe_gamma=0.01, moe_gate_floor=1e-12, moe_init=0.006, moe_kernel="scatter-dualgather", moe_round=16,
               moe_router_center=0.9, moe_router_lr_mul=0.05, moe_seq=0.001, moe_shared_frac=0.25,
               moe_update="quantile")  # fmt: skip
ALLIE21 = dict(board="conv", mlp="swiglu", key_offset=False, moe=[256, 16], moe_seq=0.001, moe_init=0.006,
               moe_router_lr_mul=0.05, moe_gamma=0.01, moe_update="quantile", moe_kernel="scatter-dualgather",
               moe_shared_frac=0.25, moe_round=16, fp32_small_masters=True, moe_log_gates=True, moe_dense_first=3,
               moe_router_center=0.9, moe_router_center_first=4, xsa=True, moe_shard=True)  # fmt: skip


def checkpoint(path, arch):
    """A training checkpoint's state (the trainer's names and layouts) that export.convert maps to the export at
    path, its attention lambdas 1; uncentred layers (zero centre) have no mu."""
    w, c = (
        load_file(path / "model.safetensors"),
        json.loads((path / "config.json").read_text()),
    )
    n, d = c["layers"], c["width"]
    s = w["scalars"].clone()
    s[n : 3 * n] = 1
    m = dict(scalars=s, x0_lambdas=w["x0_lambdas"])
    m |= {f"{k}.weight": w[k] for k in ("embed", "embed2", "lm_head", "feat_embed", "smear_gate")}
    m |= {f"skip_gates.{j}.weight": w[f"skip_gate.{j}"] for j in range(3)}
    m |= {f"value_embeds.{j}.weight": w[f"value_embed.{j}"] for j in range(c["value_embeds"])}
    m |= {k: v for k, v in w.items() if k.startswith("board.")}
    if "xsa" in w:
        m["xsa_alphas"] = w["xsa"]
    for i in range(n):
        b = f"blocks.{i}."
        m[b + "attn.qkvo_w"] = torch.cat((w[f"{i}.qkv"], w[f"{i}.o"]))
        m[b + "attn.attn_gate.weight"] = w[f"{i}.attn_gate"]
        m[b + "attn.value_embed_gate.weight"] = w[f"{i}.ve_gate"]
        if f"{i}.fc" in w:
            m[b + "mlp.c_fc"], m[b + "mlp.c_proj"] = w[f"{i}.fc"], w[f"{i}.proj"].T
            continue
        m[b + "mlp.router"], m[b + "mlp.bias"] = w[f"{i}.router"], w[f"{i}.moe_bias"]
        if w[f"{i}.mu"].any():
            m[b + "mlp.mu"], m[b + "mlp.mu_steps"] = w[f"{i}.mu"], torch.tensor(100.0)
        m[b + "mlp.up"], m[b + "mlp.down"] = (
            w[f"{i}.up"],
            w[f"{i}.down"].transpose(1, 2),
        )
        m[b + "mlp.shared_up"], m[b + "mlp.shared_down"] = (
            w[f"{i}.shared_up"],
            w[f"{i}.shared_down"].T,
        )
    arch = arch | dict(moe=[c["experts"], c["topk"]])
    yarn = dict(cos=w["cos"], sin=w["sin"], attn_scale=c["attn_scale"])
    return dict(
        config=dict(layers=n, width=d, head_dim=c["head_dim"], feats=3, arch=arch),
        model=m,
        inference=dict(split_embed=True, yarn=yarn, ws_short=11, ws_long=23),
    ), (w, c)


@pytest.mark.parametrize("arch", ["allie20", "allie21"])
def test_convert(tmp_path, arch):
    """A checkpoint of Allie 2.0's or 2.1's switches (XSA, log gates, dense_first 3, centring on the first MoE
    layers) converts to the export it came from; 2.0's has no new keys."""
    arch, kw = dict(allie20=(ALLIE20, {}), allie21=(ALLIE21, C4X | dict(centre_first=4)))[arch]
    state, (w, c) = checkpoint(tiny_export(tmp_path, **kw), arch)
    got, config = convert(state)
    n = c["layers"]
    w["scalars"][n : 3 * n] = 1
    assert got.keys() == w.keys()
    for k, v in got.items():
        assert v.is_contiguous() and torch.equal(v, w[k].to(v.dtype)), k
        assert v.dtype == (torch.float32 if k.split(".")[-1] in ("router", "moe_bias", "mu", "scalars",
                                                                 "x0_lambdas", "xsa") else torch.bfloat16)  # fmt: skip
    del c["name"]
    assert config == c | dict(gate_floor=arch.get("moe_gate_floor", 0.0))
    assert ("log_gates" in config) == ("xsa" in got) == (arch is ALLIE21)


def test_convert_refuses(tmp_path):
    state, _ = checkpoint(tiny_export(tmp_path, **C4X), ALLIE21)
    for bad in (
        dict(moe_keep=8),
        dict(xsa=False),
        dict(moe_dense_first=2),
        dict(header_feats=2),
    ):
        s = state | dict(config=state["config"] | dict(arch=state["config"]["arch"] | bad))
        with pytest.raises(AssertionError):
            convert(s)
