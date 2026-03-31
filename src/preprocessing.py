"""
Preprocessing for NFL tracking data.
Converts CSV -> (T, N, F) tensors per play.
"""

import pandas as pd
import numpy as np


def load_and_group_plays(csv_path):
    """
    Load tracking CSV and group by play.

    Returns: dict mapping (gameId, playId) -> DataFrame
    """
    df = pd.read_csv(csv_path)


    # Group by play
    plays = {}
    for (game_id, play_id), play_df in df.groupby(['gameId', 'playId']):
        frame_col = 'frameId' if 'frameId' in play_df.columns else 'frame.id'
        plays[(game_id, play_id)] = play_df.sort_values(frame_col)

    return plays


def normalize_field_direction(play_df):
    """
    Flip coordinates so offense always moves left→right.

    NFL provides 'playDirection' (left/right). When left, we flip:
    - x → 120 - x (field is 120 yards)
    - dir → (180 - dir) % 360 (flip heading angle)
    """
    play_df = play_df.copy()

    if 'playDirection' in play_df.columns:
        mask = play_df['playDirection'] == 'left'

        # Flip x-coordinate
        play_df.loc[mask, 'x'] = 120 - play_df.loc[mask, 'x']

        # Flip direction angle
        if 'dir' in play_df.columns:
            play_df.loc[mask, 'dir'] = (180 - play_df.loc[mask, 'dir']) % 360

    return play_df


def play_to_tensor(play_df, max_frames=100, max_players=22, features=['x', 'y', 's', 'dir']):
    """
    Convert single play DataFrame → (T, N, F) tensor.

    Returns:
        data: (max_frames, max_players, len(features)) tensor
        mask: (max_frames, max_players) binary mask
    """
    frame_col = 'frameId' if 'frameId' in play_df.columns else 'frame.id'
    frames = sorted(play_df[frame_col].unique())
    players = play_df['nflId'].unique() if 'nflId' in play_df.columns else list(range(len(play_df)))

    T = min(len(frames), max_frames)
    N = min(len(players), max_players)
    F = len(features)

    data = np.zeros((max_frames, max_players, F), dtype=np.float32)
    mask = np.zeros((max_frames, max_players), dtype=np.float32)

    player_to_idx = {p: i for i, p in enumerate(list(players)[:N])}

    for t, frame_id in enumerate(frames[:T]):
        frame_data = play_df[play_df[frame_col] == frame_id]
        for _, row in frame_data.iterrows():
            player_id = row.get('nflId', row.name)
            if player_id in player_to_idx:
                n = player_to_idx[player_id]
                for f, feat in enumerate(features):
                    if feat in row and pd.notna(row[feat]):
                        data[t, n, f] = row[feat]
                mask[t, n] = 1.0

    return data, mask
