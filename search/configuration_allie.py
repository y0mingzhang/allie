"""Hugging Face metadata for Allie's inference-only architecture."""
from transformers import PretrainedConfig


class AllieConfig(PretrainedConfig):
    model_type = 'allie_chess'

    def __init__(self, allie=None, **kwargs):
        self.allie = allie or {}
        c = self.allie
        defaults = dict(hidden_size=c.get('width',512), num_hidden_layers=c.get('layers',8),
                        num_attention_heads=c.get('width',512)//c.get('head_dim',64),
                        num_key_value_heads=c.get('width',512)//c.get('head_dim',64),
                        head_dim=c.get('head_dim',64), vocab_size=c.get('vocab_size',2432),
                        max_position_embeddings=1025, tie_word_embeddings=False,
                        architectures=['AllieForCausalLM'])
        super().__init__(**(defaults | kwargs))
