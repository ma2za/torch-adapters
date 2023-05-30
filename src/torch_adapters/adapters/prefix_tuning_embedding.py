import torch
from torch import nn, Tensor

from torch_adapters.adapters.mixin import AdapterMixin


def prefix_attention_mask(attention_mask: Tensor, prefix_length: int) -> Tensor:
    """

    :param attention_mask:
    :param prefix_length:
    :return:
    """
    prefix_mask = torch.ones((attention_mask.shape[0], prefix_length),
                             dtype=attention_mask.dtype,
                             device=attention_mask.device)
    return torch.cat([prefix_mask, attention_mask], dim=-1)


class PrefixTokenTypeEmbedding(nn.Embedding, AdapterMixin):
    def __init__(self, src: nn.Embedding, prefix_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        self.copy_attributes_from_source(src)
        self.prefix_length = prefix_length

    def forward(self, input: Tensor) -> Tensor:
        prefix = input[:, 0].unsqueeze(1).expand(-1, self.prefix_length)
        extended_input = torch.cat([prefix, input], dim=1)
        return super().forward(extended_input)


class PrefixAbsolutePositionalEmbedding(nn.Embedding, AdapterMixin):
    def __init__(self, src: nn.Embedding, prefix_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        self.copy_attributes_from_source(src)
        self.prefix_length = prefix_length

    def forward(self, input: Tensor) -> Tensor:
        prefix_ids = torch.arange(1, 1 + self.prefix_length, dtype=torch.long, device=input.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(input.shape[0], -1)
        mask = input.ne(self.padding_idx).int()
        incremental_indices = (input - self.padding_idx + prefix_ids.max()) * mask
        extended_input = torch.cat([prefix_ids, incremental_indices], dim=1) + self.padding_idx
        return super().forward(extended_input)


class PrefixTuningEmbedding(nn.Embedding, AdapterMixin):
    """

    Prefix-Tuning: Optimizing Continuous Prompts for Generation
    by Xiang Lisa Li, Percy Liang
    https://arxiv.org/abs/2101.00190

    """

    def __init__(self,
                 src: nn.Embedding,
                 prefix_length: int,
                 hidden_rank: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        self.copy_attributes_from_source(src)
        self.prefix_length = prefix_length

        self.prefix_embedding = nn.Embedding(self.prefix_length, self.embedding_dim)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_rank),
            nn.Tanh(),
            nn.Linear(hidden_rank, self.embedding_dim),
        )

    def forward(self, input: Tensor) -> Tensor:
        prefix_ids = torch.arange(self.prefix_length, dtype=torch.long, device=input.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(input.shape[0], -1)
        prefix = self.prefix_mlp(self.prefix_embedding(prefix_ids))
        return torch.cat([prefix, self.weight[input]], dim=1)
