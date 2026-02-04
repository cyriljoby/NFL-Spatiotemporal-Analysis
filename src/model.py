"""
Spatiotemporal encoder for NFL plays.

Architecture:
  Input: (B, T, N, F) tensor
    ↓
  Per-player MLP: Encode each player's features
    ↓
  Player pooling: Aggregate across players (mean)
    ↓
  LSTM: Capture temporal dynamics
    ↓
  Output: (B, embedding_dim) play embedding
"""

import torch
import torch.nn as nn


class SpatiotemporalEncoder(nn.Module):
    """
    Encodes a play (T, N, F) into a fixed-size embedding.

    Design decisions:
    - Per-player MLP: Each player processed independently
    - Mean pooling: Simple aggregation across players
    - LSTM: Captures temporal patterns
    - Final pooling: Mean over time (could also use last timestep)
    """

    def __init__(
        self,
        input_dim=4,           # x, y, s, dir
        player_hidden_dim=32,  # Hidden dim for per-player MLP
        lstm_hidden_dim=64,    # LSTM hidden dimension
        embedding_dim=128,     # Final embedding size
        num_lstm_layers=1      # Number of LSTM layers
    ):
        super().__init__()

        self.input_dim = input_dim
        self.player_hidden_dim = player_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_dim = embedding_dim

        # Per-player MLP: (F) -> (player_hidden_dim)
        # Processes each player's features independently
        self.player_encoder = nn.Sequential(
            nn.Linear(input_dim, player_hidden_dim),
            nn.ReLU(),
            nn.Linear(player_hidden_dim, player_hidden_dim),
            nn.ReLU()
        )

        # LSTM: Process temporal sequence
        # Input: (T, player_hidden_dim) per sample
        # Output: (T, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=player_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True  # Input is (B, T, player_hidden_dim)
        )

        # Final projection to embedding space
        self.projection = nn.Linear(lstm_hidden_dim, embedding_dim)

    def forward(self, data, mask):
        """
        Args:
            data: (B, T, N, F) tensor of player positions
            mask: (B, T, N) binary mask (1 = valid, 0 = padding)

        Returns:
            embeddings: (B, embedding_dim) play embeddings
        """
        B, T, N, F = data.shape

        # Step 1: Per-player encoding
        # Reshape to process all players together: (B*T*N, F)
        data_flat = data.reshape(B * T * N, F)
        player_features = self.player_encoder(data_flat)  # (B*T*N, player_hidden_dim)

        # Reshape back: (B, T, N, player_hidden_dim)
        player_features = player_features.reshape(B, T, N, self.player_hidden_dim)

        # Step 2: Aggregate players at each timestep
        # Use masked mean pooling to handle variable number of players
        mask_expanded = mask.unsqueeze(-1)  # (B, T, N, 1)
        masked_features = player_features * mask_expanded  # Zero out padding

        # Sum over players and divide by number of valid players
        player_sum = masked_features.sum(dim=2)  # (B, T, player_hidden_dim)
        player_count = mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, T, 1)
        timestep_features = player_sum / player_count  # (B, T, player_hidden_dim)

        # Step 3: Temporal encoding with LSTM
        lstm_out, (h_n, c_n) = self.lstm(timestep_features)  # lstm_out: (B, T, lstm_hidden_dim)

        # Step 4: Aggregate over time
        # Use mean pooling over valid timesteps
        # Create time-level mask: (B, T)
        time_mask = (mask.sum(dim=2) > 0).float()  # (B, T)
        time_mask_expanded = time_mask.unsqueeze(-1)  # (B, T, 1)

        masked_lstm = lstm_out * time_mask_expanded  # (B, T, lstm_hidden_dim)
        lstm_sum = masked_lstm.sum(dim=1)  # (B, lstm_hidden_dim)
        time_count = time_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        play_features = lstm_sum / time_count  # (B, lstm_hidden_dim)

        # Step 5: Project to embedding space
        embeddings = self.projection(play_features)  # (B, embedding_dim)

        return embeddings


class PlayAutoencoder(nn.Module):
    """
    Simple autoencoder for self-supervised learning.

    Encoder: SpatiotemporalEncoder (play -> embedding)
    Decoder: MLP (embedding -> reconstructed positions)

    Training objective: Reconstruct input play from embedding
    """

    def __init__(
        self,
        input_dim=4,
        player_hidden_dim=32,
        lstm_hidden_dim=64,
        embedding_dim=128,
        max_frames=100,
        max_players=22
    ):
        super().__init__()

        self.max_frames = max_frames
        self.max_players = max_players
        self.input_dim = input_dim

        # Encoder: Play -> Embedding
        self.encoder = SpatiotemporalEncoder(
            input_dim=input_dim,
            player_hidden_dim=player_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            embedding_dim=embedding_dim
        )

        # Decoder: Embedding -> Reconstructed play
        # Output size: T * N * F
        output_size = max_frames * max_players * input_dim

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, data, mask):
        """
        Args:
            data: (B, T, N, F) input play
            mask: (B, T, N) binary mask

        Returns:
            reconstruction: (B, T, N, F) reconstructed play
            embedding: (B, embedding_dim) play embedding
        """
        B = data.shape[0]

        # Encode
        embedding = self.encoder(data, mask)  # (B, embedding_dim)

        # Decode
        reconstruction_flat = self.decoder(embedding)  # (B, T*N*F)

        # Reshape to play format
        reconstruction = reconstruction_flat.reshape(
            B, self.max_frames, self.max_players, self.input_dim
        )  # (B, T, N, F)

        return reconstruction, embedding

    def encode(self, data, mask):
        """Just return the embedding (for inference)."""
        return self.encoder(data, mask)
