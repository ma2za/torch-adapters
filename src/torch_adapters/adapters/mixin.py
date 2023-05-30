from torch import nn


class AdapterMixin:

    def copy_attributes_from_source(self, src: nn.Module):
        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))
