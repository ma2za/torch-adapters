import inspect

import torch
from torch import nn

from torch_adapters.adapters.mixin import AdapterMixin


class PrefixTuning(AdapterMixin, nn.Module):
    """

    Prefix-Tuning: Optimizing Continuous Prompts for Generation
    by Xiang Lisa Li, Percy Liang
    https://arxiv.org/abs/2101.00190
    """

    def __init__(self, src: nn.Module, prefix_size: int, hidden_size: int):
        super().__init__()

        self.prefix_embedding = nn.Embedding(prefix_size, src.all_head_size)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(src.all_head_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 * src.all_head_size),
        )
        self.base_self = src

        self.prefix_size = prefix_size

    def drop(self):
        del self._modules["prefix_mlp"]
        self.prefix_mlp = None

    def merge_args_kwargs(self, *args, **kwargs):
        args_names = inspect.getfullargspec(self.base_self.forward).args
        if args_names[0] == "self":
            args_names.pop(0)
        args_names = args_names[: len(args)]

        prefix_kwargs = {**dict(zip(args_names, args)), **kwargs}
        return prefix_kwargs

    def forward(self, *args, **kwargs):
        all_kwargs = self.merge_args_kwargs(*args, **kwargs)

        # prefix forward
        # TODO move hardcoded names to config
        prefix_inputs = torch.arange(
            self.prefix_size,
            dtype=torch.long,
            device=all_kwargs["hidden_states"].device,
        )
        prefix_inputs = prefix_inputs.unsqueeze(0).expand(
            all_kwargs["hidden_states"].shape[0], -1
        )
        past_key_value = self.prefix_embedding(prefix_inputs)
        if self.prefix_mlp is not None:
            past_key_value = self.prefix_mlp(past_key_value)
        else:
            past_key_value = torch.cat(
                [past_key_value.unsqueeze(0), past_key_value.unsqueeze(0)], dim=0
            )
        past_key_value = past_key_value.view(
            2,
            -1,
            self.base_self.num_attention_heads,
            self.prefix_size,
            self.base_self.attention_head_size,
        )
        # prefix mask
        mask_shape = list(all_kwargs["attention_mask"].shape)
        mask_shape[-1] = self.prefix_size
        all_kwargs["attention_mask"] = torch.cat(
            [
                torch.zeros(
                    mask_shape,
                    device=all_kwargs["attention_mask"].device,
                    dtype=all_kwargs["attention_mask"].dtype,
                ),
                all_kwargs["attention_mask"],
            ],
            dim=-1,
        )
        return self.base_self(**{**all_kwargs, **{"past_key_value": past_key_value}})
