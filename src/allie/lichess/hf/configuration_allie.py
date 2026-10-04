"""Allie 2.0's configuration for transformers (trust_remote_code)."""

from transformers import PretrainedConfig


class AllieConfig(PretrainedConfig):
    """The network's shape (model.py reads it): layers, width, heads, head_dim, vocab,
    value_embeds, experts, topk, expert_hidden, shared_hidden, gate_floor, attn_scale."""

    model_type = "allie"
