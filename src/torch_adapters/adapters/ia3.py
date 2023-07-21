from torch import nn, Tensor

from torch_adapters.adapters.mixin import AdapterMixin

__all__ = ["IA3"]


class IA3(nn.Linear, AdapterMixin):

    def __init__(self, src: nn.Linear, ia3_first: bool = False):
        super().__init__(src.in_features, src.out_features)

        self.ia3_first = ia3_first

        self.copy_attributes_from_source(src)

        self.ia3_weight = nn.Linear(1, src.in_features if ia3_first else src.out_features, bias=False)

        nn.init.ones_(self.ia3_weight.weight)

    def merge(self):
        # TODO check correctness
        merged_layer = nn.Linear(self.in_features, self.out_features)
        merged_weight = self.weight.data * self.ia3_weight.weight.data if self.ia3_first else self.ia3_weight.weight.data * self.weight.data
        merged_layer.weight.data = merged_weight.detach().clone().to(self.weight.device)
        merged_layer.bias.data = self.bias.data.detach().clone().to(self.bias.device)
        return merged_layer

    def forward(self, input_ids: Tensor) -> Tensor:
        # TODO check correctness
        return super().forward(self.ia3_weight(input_ids)) if self.ia3_first else self.ia3_weight(
            super().forward(input_ids))
