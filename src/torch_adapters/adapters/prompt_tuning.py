import torch
from torch import nn, Tensor
from torch.nn import Parameter

from torch_adapters.adapters.mixin import AdapterMixin


def prompt_attention_mask(attention_mask: Tensor, prompt_length: int) -> Tensor:
    """

    :param attention_mask:
    :param prompt_length:
    :return:
    """
    prompt_mask = torch.ones((attention_mask.shape[0], prompt_length),
                             dtype=attention_mask.dtype,
                             device=attention_mask.device)
    return torch.cat([prompt_mask, attention_mask], dim=-1)


class PromptTokenTypeEmbedding(nn.Embedding, AdapterMixin):
    def __init__(self, src: nn.Embedding, prompt_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        self.copy_attributes_from_source(src)
        self.prompt_length = prompt_length

    def forward(self, input: Tensor) -> Tensor:
        prompt = input[:, 0].unsqueeze(1).expand(-1, self.prompt_length)
        extended_input = torch.cat([prompt, input], dim=1)
        return super().forward(extended_input)


class PromptAbsolutePositionalEmbedding(nn.Embedding, AdapterMixin):
    def __init__(self, src: nn.Embedding, prompt_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        self.copy_attributes_from_source(src)
        self.prompt_length = prompt_length

    def forward(self, input: Tensor) -> Tensor:
        prompt_ids = torch.arange(1, 1 + self.prompt_length, dtype=torch.long, device=input.device)
        prompt_ids = prompt_ids.unsqueeze(0).expand(input.shape[0], -1)
        mask = input.ne(self.padding_idx).int()
        incremental_indices = (input - self.padding_idx + prompt_ids.max()) * mask
        extended_input = torch.cat([prompt_ids, incremental_indices], dim=1) + self.padding_idx
        return super().forward(extended_input)


class PromptTuningEmbedding(nn.Embedding, AdapterMixin):
    """

    The Power of Scale for Parameter-Efficient Prompt Tuning
    by Brian Lester, Rami Al-Rfou and Noah Constant
    https://arxiv.org/abs/2104.08691

    """

    def __init__(self,
                 src: nn.Embedding,
                 prompt_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        self.copy_attributes_from_source(src)
        self.prompt_length = prompt_length

        self.prompt_embedding = nn.Embedding(self.prompt_length, self.embedding_dim)

        # TODO add random init, best 5000 sampling and text init and move to reset_parameters method
        self.prompt_embedding.weight = Parameter(self.weight[torch.randint(0, self.weight.shape[0], (
            self.prompt_embedding.num_embeddings,))].detach().clone())

    def forward(self, input: Tensor) -> Tensor:
        prompt_ids = torch.arange(self.prompt_length, dtype=torch.long, device=input.device)
        prompt_ids = prompt_ids.unsqueeze(0).expand(input.shape[0], -1)
        prompt = self.prompt_embedding(prompt_ids)
        return torch.cat([prompt, self.weight[input]], dim=1)
