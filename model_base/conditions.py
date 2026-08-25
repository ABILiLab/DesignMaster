from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):


    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.reshape(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ClusterContinuousEmbedder(nn.Module):


    def __init__(self, in_dim: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embedding_drop = nn.Embedding(1, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.dropout_prob = dropout_prob

    def forward(
        self,
        x: torch.Tensor,
        train: bool,
        force_drop_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.mlp(x)
        if force_drop_id is None:
            if train and self.dropout_prob > 0:
                drop_ids = torch.rand(x.shape[0], device=x.device) < self.dropout_prob
            else:
                return emb
        else:
            drop_ids = force_drop_id.view(-1).bool()
        if drop_ids.any():
            null = self.embedding_drop.weight[0].unsqueeze(0).expand_as(emb)
            emb = torch.where(drop_ids.unsqueeze(-1), null, emb)
        # NaNs (missing properties) → null embedding
        nan_mask = torch.isnan(x).any(dim=-1)
        if nan_mask.any():
            null = self.embedding_drop.weight[0].unsqueeze(0).expand_as(emb)
            emb = torch.where(nan_mask.unsqueeze(-1), null, emb)
        return emb


class CategoricalEmbedder(nn.Module):

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.dropout_prob = dropout_prob

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = labels.long().view(-1)
        if force_drop_id is None:
            if train and self.dropout_prob > 0:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            else:
                drop_ids = None
        else:
            drop_ids = force_drop_id.view(-1).bool()
        if drop_ids is not None:
            labels = torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)
        return self.embedding_table(labels)
