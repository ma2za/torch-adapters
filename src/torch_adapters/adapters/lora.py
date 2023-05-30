from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import init

from torch_adapters.adapters.mixin import AdapterMixin


class LoRA(nn.Linear, AdapterMixin):
    """

    LoRA: Low-Rank Adaptation of Large Language Models
    by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu,
    Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
    https://arxiv.org/abs/2106.09685

    """

    def __init__(self,
                 src: nn.Linear,
                 alpha: int = 8,
                 r: int = 8):
        super().__init__(src.in_features, src.out_features)

        self.copy_attributes_from_source(src)

        self.lora_weight = torch.nn.Sequential(OrderedDict([
            ("A", nn.Linear(self.in_features, r, bias=False)),
            ("B", nn.Linear(r, self.out_features, bias=False))
        ]))

        self.alpha = alpha
        self.r = r

        init.zeros_(self.lora_weight.B.weight)
        init.normal_(self.lora_weight.A.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        return super().forward(input_ids) + self.lora_weight(input_ids) * (self.alpha / self.r)
