from collections import OrderedDict

from torch import nn, Tensor

from torch_adapters.adapters.mixin import AdapterMixin


class LoRA(nn.Linear, AdapterMixin):
    """

    Layers to train at finetuning: adapter, layer-norm, classifier.

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

        self.lora_weight = nn.Sequential(OrderedDict([
            ("A", nn.Linear(self.in_features, r, bias=False)),
            ("B", nn.Linear(r, self.out_features, bias=False))
        ]))

        self.alpha = alpha
        self.r = r

        nn.init.zeros_(self.lora_weight.B.weight)
        nn.init.normal_(self.lora_weight.A.weight)

    def merge(self) -> nn.Linear:
        # TODO add copy of other attributes
        # TODO check if matrix transpose is required
        merged_layer = nn.Linear(self.in_features, self.out_features)
        merged_weight = self.weight.data + (self.alpha / self.r) * (
                self.lora_weight.B.weight.data @ self.lora_weight.A.weight.data)
        merged_layer.weight.data = merged_weight.detach().clone().to(self.weight.device)
        merged_layer.bias.data = self.bias.data.detach().clone().to(self.bias.device)
        return merged_layer

    def forward(self, input_ids: Tensor) -> Tensor:
        return super().forward(input_ids) + self.lora_weight(input_ids) * (self.alpha / self.r)
