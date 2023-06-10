from collections import OrderedDict

from torch import Tensor
from torch.nn import GELU, init, Linear, Module, Sequential

from torch_adapters.adapters.mixin import AdapterMixin

__all__ = ["Adapter"]


class Adapter(AdapterMixin, Module):
    """

    Parameter-Efficient Transfer Learning for NLP
    by Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone,
    Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly

    """

    def __init__(self, src: Linear, adapter_size: int):
        super().__init__()
        self.src = src
        self.adapter = Sequential(
            OrderedDict(
                [
                    ("A", Linear(src.out_features, adapter_size, bias=True)),
                    ("act", GELU()),
                    ("B", Linear(adapter_size, src.out_features, bias=True)),
                ]
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        # TODO check if gaussian init is the standard
        for param in self.adapter.parameters():
            init.zeros_(param)

    def forward(self, input: Tensor) -> Tensor:
        output = self.src(input)
        return self.adapter(output) + output
