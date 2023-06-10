from torch.nn import Module


class AdapterMixin:
    def copy_attributes_from_source(self, src: Module):
        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))
