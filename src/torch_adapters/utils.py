from operator import attrgetter
from typing import List, Dict

import torch
from torch import nn

from .adapters.lora import LoRA
from .adapters.prefix_tuning_embedding import PrefixTuningEmbedding, PrefixTokenTypeEmbedding, \
    PrefixAbsolutePositionalEmbedding


# TODO group in una utility class

# TODO consider if unify in one add method with configuration


def add_lora(model: nn.Module, layers_names: List[str], config: Dict) -> torch.nn.Module:
    """

    Replace in-place the linear layers named in layers_names with a LoRA layer
    having the same weight and bias parameters.

    :param model:
    :param layers_names:
    :param config:
    :return:
    """
    for name, module in model.named_modules():
        if any([i in name for i in layers_names]):
            module_name, attr_name = name.rsplit(".", 1)
            # TODO check if this check must be added
            # if attr_name not in layers_names:
            #   continue
            module: torch.nn.Module = attrgetter(module_name)(model)
            attr: torch.nn.Linear = attrgetter(name)(model)
            module.__setattr__(attr_name, LoRA(
                attr,
                alpha=config.get("alpha", 8),
                r=config.get("r", 8)
            ))
    return model


def add_prefix_tuning_embedding(model: nn.Module, embeddings: Dict, config: Dict) -> nn.Module:
    for name, module in model.named_modules():
        if any([i in name for i in embeddings.keys()]):
            module_name, attr_name = name.rsplit(".", 1)
            if attr_name not in embeddings.keys():
                continue
            module: nn.Module = attrgetter(module_name)(model)
            attr: nn.Embedding = attrgetter(name)(model)
            embedding_type = embeddings.get(attr_name)
            extended_embedding = None
            if embedding_type == "word":
                extended_embedding = PrefixTuningEmbedding(src=attr,
                                                           prefix_length=config.get("prefix_length", 30),
                                                           hidden_rank=config.get("hidden_rank"))
            elif embedding_type == "token_type":
                extended_embedding = PrefixTokenTypeEmbedding(src=attr,
                                                              prefix_length=config.get("prefix_length", 30))
            elif embedding_type == "position":
                # TODO check relative embeddings
                extended_embedding = PrefixAbsolutePositionalEmbedding(src=attr,
                                                                       prefix_length=config.get("prefix_length", 30))
            if extended_embedding is None:
                # TODO replace with custom exception
                raise Exception
            module.__setattr__(attr_name, extended_embedding)
    return model


def train_adapters(model: nn.Module, names: List[str]):
    for name, param in model.named_parameters():
        param.requires_grad_(any([i in name for i in names]))
