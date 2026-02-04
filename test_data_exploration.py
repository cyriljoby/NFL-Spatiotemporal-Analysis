"""Quick test of preprocessing with real data."""

import sys
sys.path.append('src')

from preprocessing import load_and_group_plays, normalize_field_direction, play_to_tensor

# Test 1: Load and group plays
print("=" * 50)
print("TEST: Load and group plays")
print("=" * 50)

plays = load_and_group_plays('data/tracking_gameId_2017090700.csv')
print(f"Loaded {len(plays)} plays")

# Test 2: Pick one play and convert to tensor
play_id = list(plays.keys())[0]
sample_play = plays[play_id]

print(f"\nSample play {play_id}:")
print(f"  Frames: {sample_play['frame.id'].nunique()}")
print(f"  Players: {sample_play['nflId'].nunique()}")

# Test 3: Normalize (won't do anything since no playDirection column)
normalized = normalize_field_direction(sample_play)

# Test 4: Convert to tensor
print("\nConverting to tensor...")
data, mask = play_to_tensor(normalized, max_frames=100, max_players=22)

print(f"Tensor shape: {data.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Valid entries: {mask.sum():.0f}")
print(f"Active frames: {(mask.sum(axis=1) > 0).sum()}")

print("\n✓ All preprocessing functions work!")
