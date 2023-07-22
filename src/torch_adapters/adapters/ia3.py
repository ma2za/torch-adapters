import torch
from torch import nn, Tensor

from torch_adapters.adapters.mixin import AdapterMixin

__all__ = ["IA3"]


class IA3(nn.Linear, AdapterMixin):

    def __init__(self, src: nn.Linear, ia3_first: bool = False):
        super().__init__(src.in_features, src.out_features)

        self.ia3_first = ia3_first

        self.copy_attributes_from_source(src)

        self.ia3_weight = nn.Parameter(torch.empty(src.in_features if ia3_first else src.out_features))

        nn.init.ones_(self.ia3_weight)

    def merge(self):
        # TODO check correctness
        merged_layer = nn.Linear(self.in_features, self.out_features)
        merged_weight = self.weight.data * self.ia3_weight if self.ia3_first else self.ia3_weight * self.weight.data
        merged_layer.weight.data = merged_weight.detach().clone().to(self.weight.device)
        merged_layer.bias.data = self.bias.data.detach().clone().to(self.bias.device)
        return merged_layer

    def forward(self, input_ids: Tensor) -> Tensor:
        # TODO check for the bias
        return super().forward(input_ids * self.ia3_weight) if self.ia3_first else super().forward(
            input_ids) * self.ia3_weight
