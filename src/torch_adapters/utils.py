import re
from operator import attrgetter
from typing import List, Dict

import torch
from torch import nn

from .adapters.adapter import Adapter
from .adapters.ia3 import IA3
from .adapters.prefix_tuning import PrefixTuning
from .adapters.prompt_tuning import (
    PromptTuningEmbedding,
    PromptTokenTypeEmbedding,
    PromptAbsolutePositionalEmbedding,
)


# TODO group in a utility class

# TODO consider if unify in one add method with configuration


def add_prefix_tuning(
        model: nn.Module, layers_names: List[str], config: Dict
) -> torch.nn.Module:
    for name, module in model.named_modules():
        if any([i in name for i in layers_names]):
            module_name, attr_name = name.rsplit(".", 1)
            module: nn.Module = attrgetter(module_name)(model)
            if attr_name not in layers_names:
                continue
            attr: nn.Module = attrgetter(name)(model)
            module.__setattr__(
                attr_name,
                PrefixTuning(
                    attr,
                    prefix_size=config.get("prefix_size", 64),
                    hidden_size=config.get("prefix_size", 768),
                ),
            )
    return model


def drop_prefix_tuning_reparametrization(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, PrefixTuning):
            module.drop()
    return model


def add_adapter(
        model: nn.Module, layers_names: List[str], config: Dict
) -> torch.nn.Module:
    """

    :param model:
    :param layers_names:
    :param config:
    :return:
    """
    for name, module in model.named_modules():
        if any([i in name for i in layers_names]):
            module_name, attr_name = name.rsplit(".", 1)
            module: nn.Module = attrgetter(module_name)(model)
            attr: nn.Module = attrgetter(name)(model)

            # TODO specialize exception
            if not isinstance(attr, nn.Linear):
                raise Exception

            module.__setattr__(
                attr_name, Adapter(attr, adapter_size=config.get("adapter_size", 64))
            )
    return model


def add_ia3(model: nn.Module, layers_names: Dict[str, bool]) -> torch.nn.Module:
    """

    :param model:
    :param layers_names: if it set to True than it first scales and then it
    :return:
    """
    for name, module in model.named_modules():
        print(name)
        if any([re.search(i, name) for i in layers_names.keys()]):
            module_name, attr_name = name.rsplit(".", 1)
            if attr_name not in layers_names:
                continue
            module: nn.Module = attrgetter(module_name)(model)
            attr: nn.Linear = attrgetter(name)(model)

            # TODO specialize exception
            if not isinstance(attr, nn.Linear):
                raise Exception

            module.__setattr__(
                attr_name,
                IA3(attr),
            )
    return model


def merge_ia3(model):
    for name, module in model.named_modules():
        if isinstance(module, IA3):
            module_name, attr_name = name.rsplit(".", 1)
            parent_module: nn.Module = attrgetter(module_name)(model)
            parent_module.__setattr__(attr_name, module.merge())
    return model


def add_prompt_tuning(model: nn.Module, embeddings: Dict, config: Dict) -> nn.Module:
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
                extended_embedding = PromptTuningEmbedding(
                    src=attr, prompt_length=config.get("prompt_length", 30)
                )
            elif embedding_type == "token_type":
                extended_embedding = PromptTokenTypeEmbedding(
                    src=attr, prompt_length=config.get("prompt_length", 30)
                )
            elif embedding_type == "position":
                # TODO check relative embeddings
                extended_embedding = PromptAbsolutePositionalEmbedding(
                    src=attr, prompt_length=config.get("prompt_length", 30)
                )
            if extended_embedding is None:
                # TODO replace with custom exception
                raise Exception
            module.__setattr__(attr_name, extended_embedding)
    return model


def train_adapters(model: nn.Module, names: List[str]):
    for name, param in model.named_parameters():
        param.requires_grad_(any([i in name for i in names]))
