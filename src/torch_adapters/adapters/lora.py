import re
from collections import OrderedDict
from operator import attrgetter
from typing import List, Dict

from torch import nn, Tensor

from torch_adapters.adapters.mixin import AdapterMixin

__all__ = ["LoRA"]


class LoRA(nn.Linear, AdapterMixin):
    """

    Layers to train at finetuning: adapter, layer-norm, classifier.

    LoRA: Low-Rank Adaptation of Large Language Models
    by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu,
    Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
    https://arxiv.org/abs/2106.09685

    """

    def __init__(
            self, src: nn.Linear, alpha: int = 8, r: int = 8, dropout: float = 0.0
    ):
        super().__init__(src.in_features, src.out_features)

        self.copy_attributes_from_source(src)

        # TODO check dropout location
        self.lora_weight = nn.Sequential(
            OrderedDict(
                [
                    ("A", nn.Linear(self.in_features, r, bias=False)),
                    ("dropout", nn.Dropout(p=dropout)),
                    ("B", nn.Linear(r, self.out_features, bias=False)),
                ]
            )
        )

        self.alpha = alpha
        self.r = r

        nn.init.zeros_(self.lora_weight.B.weight)
        nn.init.normal_(self.lora_weight.A.weight)

    def merge(self) -> nn.Linear:
        # TODO add copy of other attributes
        # TODO check if matrix transpose is required
        merged_layer = nn.Linear(self.in_features, self.out_features)
        merged_weight = self.weight.data + (self.alpha / self.r) * (
                self.lora_weight.B.weight.data @ self.lora_weight.A.weight.data
        )
        merged_layer.weight.data = merged_weight.detach().clone().to(self.weight.device)
        merged_layer.bias.data = self.bias.data.detach().clone().to(self.bias.device)
        return merged_layer

    @classmethod
    def add_to_model(cls, model: nn.Module, layers_names: List[str], config: Dict) -> nn.Module:
        """

        Replace in-place the linear layers named in layers_names with a LoRA layer
        having the same weight and bias parameters.

        :param model:
        :param layers_names:
        :param config:
        :return:
        """
        for name, module in model.named_modules():
            if any([re.search(i, name) for i in layers_names]):
                module_name, attr_name = name.rsplit(".", 1)
                if attr_name not in layers_names:
                    continue
                module: nn.Module = attrgetter(module_name)(model)
                attr: nn.Linear = attrgetter(name)(model)

                # TODO specialize exception
                if not isinstance(attr, nn.Linear):
                    raise Exception

                module.__setattr__(
                    attr_name,
                    cls(attr,
                        alpha=config.get("alpha", 8),
                        r=config.get("r", 8),
                        dropout=config.get("dropout", 0.0)),
                )
        return model

    @staticmethod
    def merge_in_model(cls, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, cls):
                module_name, attr_name = name.rsplit(".", 1)
                parent_module: nn.Module = attrgetter(module_name)(model)
                parent_module.__setattr__(attr_name, module.merge())
        return model

    def forward(self, input_ids: Tensor) -> Tensor:
        return super().forward(input_ids) + self.lora_weight(input_ids) * (
                self.alpha / self.r
        )
