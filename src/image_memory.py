"""Bound temporary feed-forward activations without changing image tokens."""

import torch
from torch import nn


class ChunkedFeedForward(nn.Module):
    def __init__(self, module, chunk_size):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("Feed-forward chunk size must be positive")
        self.module = module
        self.chunk_size = chunk_size

    def forward(self, hidden_states):
        if hidden_states.shape[1] <= self.chunk_size:
            return self.module(hidden_states)
        return torch.cat([
            self.module(chunk) for chunk in hidden_states.split(self.chunk_size, dim=1)
        ], dim=1)
