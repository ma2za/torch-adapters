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
                 src: nn.Linear,
                 alpha: int = 8,
                 r: int = 8):
        super().__init__(src.in_features, src.out_features)

        # TODO maybe create utility method for this repeated code
        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))

        self.delta_weight = torch.nn.Sequential(OrderedDict([
            ("A", nn.Linear(self.in_features, r, bias=False)),
            ("B", nn.Linear(r, self.out_features, bias=False))
        ]))

        self.alpha = alpha
        self.r = r

        init.zeros_(self.delta_weight.B.weight)
        init.normal_(self.delta_weight.A.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        return super().forward(input_ids) + self.delta_weight(input_ids) * (self.alpha / self.r)
