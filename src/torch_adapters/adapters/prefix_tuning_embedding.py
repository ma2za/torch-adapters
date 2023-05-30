import torch
from torch import nn, Tensor


class PrefixTokenTypeEmbedding(nn.Embedding):
    def __init__(self, src: nn.Embedding, prefix_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))

        self.prefix_length = prefix_length

    def forward(self, input: Tensor) -> Tensor:
        extended_input = input.expand(-1, input.shape[-1] + self.prefix_length)
        return super().forward(extended_input)


class PrefixAbsolutePositionalEmbedding(nn.Embedding):
    def __init__(self, src: nn.Embedding, prefix_length: int):
        super().__init__(num_embeddings=src.num_embeddings,
                         embedding_dim=src.embedding_dim)

        for attr in vars(src).keys():
            setattr(self, attr, getattr(src, attr))

        self.prefix_length = prefix_length

    def forward(self, input: Tensor) -> Tensor:
        extended_input = input.expand(-1, input.shape[-1] + self.prefix_length)
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

    def forward(self, batch_size):
        input_ids = torch.arange(self.prefix_length, dtype=torch.long, device=self.embedding.device)
        input_ids = input_ids.unsqueeze(0).expand(batch_size, -1)
        return self.trainable_matrix(input_ids)
