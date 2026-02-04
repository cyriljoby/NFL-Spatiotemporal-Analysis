"""
PyTorch Dataset for NFL play tensors.
"""

import torch
from torch.utils.data import Dataset
from preprocessing import load_and_group_plays, normalize_field_direction, play_to_tensor


class NFLPlayDataset(Dataset):
    """
    Dataset that yields (T, N, F) tensors for each play.

    Usage:
        dataset = NFLPlayDataset('data/tracking.csv')
        play = dataset[0]  # Returns dict with 'data', 'mask', 'play_id'
    """

    def __init__(self, csv_path, max_frames=100, max_players=22, features=['x', 'y', 's', 'dir']):
        """
        Args:
            csv_path: Path to tracking CSV file
            max_frames: Max sequence length
            max_players: Max number of players
            features: List of feature columns to extract
        """
        self.max_frames = max_frames
        self.max_players = max_players
        self.features = features

        # Load all plays and convert to tensors
        print(f"Loading plays from {csv_path}...")
        plays_dict = load_and_group_plays(csv_path)

        self.plays = []
        self.play_ids = []

        for play_id, play_df in plays_dict.items():
            # Normalize field direction
            play_df = normalize_field_direction(play_df)

            # Convert to tensor
            data, mask = play_to_tensor(
                play_df,
                max_frames=max_frames,
                max_players=max_players,
                features=features
            )

            self.plays.append((data, mask))
            self.play_ids.append(play_id)

        print(f"Loaded {len(self.plays)} plays")

    def __len__(self):
        """Return number of plays."""
        return len(self.plays)

    def __getitem__(self, idx):
        """
        Get a single play.

        Returns:
            dict with keys:
                - 'data': (T, N, F) FloatTensor
                - 'mask': (T, N) FloatTensor
                - 'play_id': tuple (gameId, playId)
        """
        data, mask = self.plays[idx]

        return {
            'data': torch.FloatTensor(data),
            'mask': torch.FloatTensor(mask),
            'play_id': self.play_ids[idx]
        }
