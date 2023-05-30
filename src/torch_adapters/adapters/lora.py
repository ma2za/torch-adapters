from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import init


class LoRA(nn.Linear):
    """

    LoRA: Low-Rank Adaptation of Large Language Models
    by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu,
    Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
    https://arxiv.org/abs/2106.09685

    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 weight: Tensor,
                 bias: Tensor = None,
                 alpha: int = 8,
                 r: int = 8):
        super().__init__(in_features, out_features, bias is not None)
        self.delta_weight = torch.nn.Sequential(OrderedDict([
            ("A", nn.Linear(in_features, r, bias=False)),
            ("B", nn.Linear(r, out_features, bias=False))
        ]))

        self.alpha = alpha
        self.r = r

        self.weight = weight
        self.bias = bias

        init.zeros_(self.delta_weight.B.weight)
        init.normal_(self.delta_weight.A.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        return super().forward(input_ids) + self.delta_weight(input_ids) * (self.alpha / self.r)
