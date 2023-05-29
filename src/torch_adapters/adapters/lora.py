from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import init


class LoRA(nn.Linear):
    def __init__(self, in_features: int, out_features: int, weight: Tensor,
                 bias: Tensor, alpha: int, r: int):
        super().__init__(in_features, out_features, bias is not None)
        self.delta_weight = torch.nn.Sequential(OrderedDict([
            ("A", nn.Linear(in_features, 8, bias=False)),
            ("B", nn.Linear(8, out_features, bias=False))
        ]))

        self.alpha = alpha
        self.r = r

        init.zeros_(self.delta_weight.B.weight)
        init.normal_(self.delta_weight.A.weight)

        self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input) + self.delta_weight(input) * (self.alpha / self.r)
