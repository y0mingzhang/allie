from typing import Any, Optional, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimum.bettertransformer import BetterTransformer
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, PreTrainedModel
from transformers.utils import ModelOutput

from modeling.data import UCITokenizer


class CausalLMWithControlToken(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
    ):
        super().__init__()
        self.config = model.config
        self.model = model
        self.vocab_size = self.model.transformer.wte.num_embeddings
        embedding_dim = model.config.n_embd
        self.control_embeds = nn.Embedding(2, embedding_dim)
        self.control_embeds.weight.data.normal_(mean=0.0, std=0.02)

        self.elo_min, self.elo_max = 500, 3000

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: bool = True,
        **kwargs: dict[str, Any],
    ):
        elo_mask = (input_ids >= self.vocab_size).unsqueeze(-1)
        elo_normed = (
            (input_ids - self.vocab_size).clamp(self.elo_min, self.elo_max)
            - self.elo_min
        ) / (self.elo_max - self.elo_min)
        control_values = torch.stack((elo_normed, 1 - elo_normed), dim=2).to(
            self.control_embeds.weight.dtype
        )
        control_embeds = control_values @ self.control_embeds.weight
        token_embeds = self.model.transformer.wte(
            input_ids.clamp(0, self.vocab_size - 1)
        )

        inputs_embeds = torch.where(elo_mask, control_embeds, token_embeds)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        return outputs


class CausalLMWithRegressionHead(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        lm_loss_coef: float = 1.0,
        value_loss_coef: float = 1.0,
        time_loss_coef: float = 1.0,
    ):
        super().__init__()
        self.config = model.config
        self.model = model
        embedding_dim = model.config.n_embd
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten(start_dim=1),
            nn.Tanh(),
        )
        self.time_head = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten(start_dim=1),
        )
        self.lm_loss_coef = lm_loss_coef
        self.value_loss_coef = value_loss_coef
        self.time_loss_coef = time_loss_coef

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        value_labels: Optional[torch.LongTensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        output_hidden_states: bool = True,
        **kwargs: dict[str, Any],
    ) -> ModelOutput:
        assert output_hidden_states
        outputs = {
            k: v
            for k, v in self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            ).items()
        }

        last_hidden_states = outputs["hidden_states"][-1]
        value_preds = self.value_head(last_hidden_states)
        outputs["value_logits"] = value_preds

        time_preds = self.time_head(last_hidden_states)
        outputs["time_logits"] = time_preds

        if value_labels is not None:
            losses = F.mse_loss(
                value_preds,
                value_labels.to(device=value_preds.device, dtype=value_preds.dtype),
                reduction="none",
            )
            if torch.eq(value_labels, -100).all():
                value_loss = (losses * 0.0).mean()
            else:
                value_loss = losses[value_labels != -100].mean()
            outputs["value_loss"] = value_loss

        if time_labels is not None:
            losses = F.mse_loss(
                time_preds,
                time_labels.to(device=time_preds.device, dtype=time_preds.dtype),
                reduction="none",
            )
            if torch.eq(time_labels, -100).all():
                time_loss = (losses * 0.0).mean()
            else:
                time_loss = losses[time_labels != -100].mean()
            outputs["time_loss"] = time_loss

        loss = torch.tensor(0.0, dtype=torch.float, device=value_preds.device)

        if "loss" in outputs:
            outputs["lm_loss"] = outputs["loss"]
            loss = loss + outputs["lm_loss"] * self.lm_loss_coef
        if "value_loss" in outputs:
            loss = loss + outputs["value_loss"] * self.value_loss_coef
        if "time_loss" in outputs:
            loss = loss + outputs["time_loss"] * self.value_loss_coef

        outputs["loss"] = loss
        return outputs


Model: TypeAlias = (
    PreTrainedModel | CausalLMWithRegressionHead | CausalLMWithControlToken
)


def initialize_model(
    tokenizer: UCITokenizer,
    base_model: str,
    use_control_token: bool = False,
    use_regression_head: bool = False,
    use_pretrained: bool = False,
    lm_loss_coef: float = 1.0,
    value_loss_coef: float = 1.0,
    **kwargs: dict[str, Any],
) -> Model:
    config = AutoConfig.from_pretrained(
        base_model,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )

    model = AutoModelForCausalLM.from_config(config)
    if use_pretrained:
        # skip loading embeddings
        assert isinstance(config, GPT2Config)
        ignore_keys = ["transformer.wte.weight", "lm_head.weight"]
        pretrained_model = AutoModelForCausalLM.from_pretrained(base_model)
        state_dict = {
            k: v
            for k, v in pretrained_model.state_dict().items()
            if k not in ignore_keys
        }
        model.load_state_dict(state_dict, strict=False)
        wte_mean = pretrained_model.transformer.wte.weight.mean().item()
        wte_std = pretrained_model.transformer.wte.weight.std().item()
        torch.nn.init.normal_(model.transformer.wte.weight, wte_mean, wte_std)

    model = BetterTransformer.transform(model)

    if use_control_token:
        model = CausalLMWithControlToken(model)

    if use_regression_head:
        model = CausalLMWithRegressionHead(model, lm_loss_coef, value_loss_coef)

    return model
