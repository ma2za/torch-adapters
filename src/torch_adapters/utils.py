from operator import attrgetter
from typing import List

import torch

from adapters.lora import LoRA


def add_lora(model: torch.nn.Module, layers_names: List[str]) -> torch.nn.Module:
    """

    :param model:
    :param layers_names:
    :return:
    """
    for name, module in model.named_modules():
        if any([i in name for i in layers_names]):
            module_name, attr_name = name.rsplit(".", 1)
            module: torch.nn.Module = attrgetter(module_name)(model)
            attr: torch.nn.Linear = attrgetter(name)(model)
            module.__setattr__(attr_name, LoRA(
                in_features=attr.in_features,
                out_features=attr.out_features,
                weight=attr.weight,
                bias=attr.bias,
                alpha=8,
                r=8
            ))
    return model
