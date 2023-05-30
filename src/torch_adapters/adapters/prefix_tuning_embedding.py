import torch
from torch import nn, Tensor


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


class PrefixTokenTypeEmbedding(nn.Embedding):
    def __init__(self, src: nn.Embedding, prefix_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))

        self.prefix_length = prefix_length

    def forward(self, input: Tensor) -> Tensor:
        extended_input = torch.cat([input[:, 0].unsqueeze(1).expand(-1, self.prefix_length), input],
                                   dim=1)
        return super().forward(extended_input)


class PrefixAbsolutePositionalEmbedding(nn.Embedding):
    def __init__(self, src: nn.Embedding, prefix_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))

        self.prefix_length = prefix_length

    def forward(self, input: Tensor) -> Tensor:
        prefix_ids = torch.arange(1, 1 + self.prefix_length, dtype=torch.long, device=input.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(input.shape[0], -1)
        mask = input.ne(self.padding_idx).int()
        incremental_indices = (input - self.padding_idx + prefix_ids.max()) * mask
        extended_input = torch.cat([prefix_ids, incremental_indices], dim=1) + self.padding_idx
        return super().forward(extended_input)


class PrefixTuningEmbedding(nn.Embedding):
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

        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))

        self.prefix_length = prefix_length
        self.trainable_matrix = nn.Embedding(self.prefix_length, self.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_rank),
            nn.Tanh(),
            nn.Linear(hidden_rank, self.embedding_dim),
        )

    def forward(self, input: Tensor) -> Tensor:
        prefix_ids = torch.arange(self.prefix_length, dtype=torch.long, device=input.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(input.shape[0], -1)
        return torch.cat([self.mlp(self.trainable_matrix(prefix_ids)), self.weight[input]], dim=1)
