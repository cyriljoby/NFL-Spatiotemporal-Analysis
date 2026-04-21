"""
Transformer-based spatiotemporal encoder for NFL plays.

Key improvements over LSTM version:
  1. Player attention (replaces mean pooling) — learns which players matter
  2. Temporal self-attention (replaces LSTM) — sees all frames at once
  3. Positional encoding — tells the model "this is frame 5 vs frame 50"
"""

import torch
import torch.nn as nn
import math


class PlayerAttentionPooling(nn.Module):
    """
    Replaces mean pooling over players.

    Mean pooling:  output = (p1 + p2 + ... + p22) / 22
    Attention:     output = w1*p1 + w2*p2 + ... + w22*p22
                   where w1..w22 are learned and sum to 1
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        #PyTorch's built-in attention that computes
        # Score = query * each player key: which players matter?
        # Weight: converts raw score into probababilites. High scores get high weights, low get lower
        # Output: weighted sum of value vectors
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # A learnable query that asks "summarize these players"
        # Through training query learns to produce accurate weights fro players
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, player_features, mask):
        """
        Args:
            player_features: (B, T, N, D) — all players at all timesteps
            mask: (B, T, N) — valid player mask

        Returns:
            timestep_features: (B, T, D) — one vector per timestep
        """
        B, T, N, D = player_features.shape

        # Reshape: treat each (batch, timestep) as an independent sequence of N players
        features = player_features.reshape(B * T, N, D)

        # Expand query for every (batch, timestep) pair
        query = self.query.expand(B * T, 1, D)

        # Build attention mask: True = ignore this position
        key_padding_mask = ~(mask.reshape(B * T, N).bool())

        # Attend: query asks "which players matter?", keys/values are the players
        attended, _ = self.attention(
            query=query,
            key=features,
            value=features,
            key_padding_mask=key_padding_mask
        )

        # Reshape back: (B*T, 1, D) → (B, T, D)
        return attended.reshape(B, T, D)


class PositionalEncoding(nn.Module):
    """
    Adds position information to the temporal sequence.

    The problem: Transformers process all frames in parallel, so they
    have no idea if a frame is first or last. Unlike LSTM which processes
    sequentially (frame 1 → 2 → 3), a Transformer sees all frames at once
    with no ordering.

    The fix: Add a unique pattern to each frame's features that encodes
    its position. Frame 0 gets one pattern, frame 50 gets a different one.

    Uses sin/cos waves at different frequencies — each position gets a
    unique "fingerprint" that the model can use to understand time.
    """

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # register_buffer: saved with model but not a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, T, D) → adds positional encoding to each timestep"""
        return x + self.pe[:, :x.size(1), :]
