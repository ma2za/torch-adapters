from collections import OrderedDict

import torch.nn.init
from torch import nn, Tensor
from torch.nn import GELU

from torch_adapters.adapters.mixin import AdapterMixin


class Adapter(AdapterMixin, nn.Module):
    """

    Parameter-Efficient Transfer Learning for NLP
    by Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone,
    Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly

    """

    def __init__(self,
                 src: nn.Linear,
                 adapter_size: int):
        super().__init__()
        self.src = src
        self.adapter = nn.Sequential(OrderedDict([
            ("A", nn.Linear(src.out_features, adapter_size, bias=True)),
            ("act", GELU()),
            ("B", nn.Linear(adapter_size, src.out_features, bias=True))
        ]))

    def reset_parameters(self):
        # TODO check if gaussian init is the standard
        for param in self.adapter.parameters():
            torch.nn.init.zeros_(param)

    def forward(self, input: Tensor) -> Tensor:
        output = self.src(input)
        return self.adapter(output) + output
