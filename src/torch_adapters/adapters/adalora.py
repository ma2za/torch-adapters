from collections import OrderedDict

from torch import nn, Tensor

from torch_adapters.adapters.mixin import AdapterMixin

__all__ = ["AdaLoRA"]


class AdaLoRA(nn.Linear, AdapterMixin):
    """

    Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
    by Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao
    https://arxiv.org/abs/2303.10512
    """

    def __init__(
            self, src: nn.Linear, alpha: int = 8, r: int = 8, dropout: float = 0.0
    ):
        super().__init__(src.in_features, src.out_features)

        self.copy_attributes_from_source(src)

        # TODO check dropout location
        self.adalora_weight = nn.Sequential(
            OrderedDict(
                [
                    ("Q", nn.Linear(self.in_features, r, bias=False)),
                    # TODO add dropout("dropout", nn.Dropout(p=dropout)),
                    ("Lambda", nn.Linear(r, r, bias=False)),  # TODO consider vector repr
                    ("P", nn.Linear(r, self.out_features, bias=False)),
                ]
            )
        )

        self.alpha = alpha
        self.r = r

        nn.init.zeros_(self.adalora_weight.Lambda.weight)
        nn.init.normal_(self.adalora_weight.Q.weight)
        nn.init.normal_(self.adalora_weight.P.weight)

    def merge(self) -> nn.Linear:
        # TODO to implement
        pass

    def forward(self, input_ids: Tensor) -> Tensor:
        # TODO to implement
        pass
