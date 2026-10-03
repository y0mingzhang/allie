import json

import pytest
import torch
from safetensors.torch import save_file

from allie.lichess.model import Model
from allie.lichess.tokens import CONTEXT


def tiny_export(path, layers=6, width=64, head_dim=16, experts=8, topk=4, seed=0):
    """A random model in export.py's layout, small enough for CPU unit tests."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, std=0.1: torch.randn(*s, generator=g) * std
    heads, ve, eh, sh, dense, vocab = (
        width // head_dim,
        min(5, layers // 2),
        8,
        16,
        32,
        2432,
    )
    freq = (1 / 1024) ** torch.linspace(0, 1, head_dim // 4)
    freq = torch.cat((freq, freq.new_zeros(head_dim // 4)))
    theta = torch.outer(torch.arange(CONTEXT, dtype=torch.float32), freq)
    w = dict(
        embed=r(vocab, width, std=1),
        embed2=r(vocab, width, std=1),
        lm_head=r(vocab, width),
        feat_embed=r(64, width),
        smear_gate=r(1, 16),
        scalars=torch.cat((1 + r(layers), 0.5 + r(2 * layers), r(5), torch.ones(3))),
        x0_lambdas=r(2 * layers),
        cos=theta.cos(),
        sin=theta.sin(),
        **{f"skip_gate.{j}": r(1, 16) for j in range(3)},
        **{f"value_embed.{j}": r(vocab, width) for j in range(ve)},
        **{"board.first": r(32, 13, 3, 3), "board.squeeze": r(8, 32, 1, 1)},
        **{f"board.residual.{j}": r(32, 32, 3, 3) for j in range(2)},
        **{"board.meta": r(32, 32), "board.output": r(544, width)},
    )
    for i in range(layers):
        w[f"{i}.qkv"], w[f"{i}.o"] = r(3 * width, width), r(width, width)
        w[f"{i}.attn_gate"], w[f"{i}.ve_gate"] = r(heads, 16), r(heads, 16)
        if i == 0:
            w["0.fc"], w["0.proj"] = r(2 * dense, width), r(width, dense)
            continue
        w[f"{i}.router"], w[f"{i}.moe_bias"], w[f"{i}.mu"] = (
            r(experts, width),
            r(experts),
            r(width),
        )
        w[f"{i}.up"], w[f"{i}.down"] = r(experts, 2 * eh, width), r(experts, width, eh)
        w[f"{i}.shared_up"], w[f"{i}.shared_down"] = r(2 * sh, width), r(width, sh)
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in w.items()}, str(path / "model.safetensors")
    )
    config = dict(layers=layers, width=width, heads=heads, head_dim=head_dim, vocab=vocab,
                  value_embeds=ve, experts=experts, topk=topk, expert_hidden=eh,
                  shared_hidden=sh, gate_floor=1e-12, attn_scale=0.1, name="tiny")  # fmt: skip
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture(scope="session")
def tiny_path(tmp_path_factory):
    return tiny_export(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def tiny(tiny_path):
    return Model(tiny_path, dtype=torch.float32)
