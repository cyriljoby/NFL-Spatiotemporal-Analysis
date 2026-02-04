"""Test the PyTorch Dataset."""

import sys
sys.path.append('src')

from dataset import NFLPlayDataset
from torch.utils.data import DataLoader

# Test 1: Create dataset
print("=" * 50)
print("TEST 1: Create Dataset")
print("=" * 50)

dataset = NFLPlayDataset('data/tracking_gameId_2017090700.csv')

print(f"Dataset size: {len(dataset)}")

# Test 2: Get a single play
print("\n" + "=" * 50)
print("TEST 2: Get single play")
print("=" * 50)

play = dataset[0]

print(f"Keys: {play.keys()}")
print(f"Data shape: {play['data'].shape}")
print(f"Mask shape: {play['mask'].shape}")
print(f"Play ID: {play['play_id']}")
print(f"Data type: {play['data'].dtype}")

# Test 3: DataLoader (batching)
print("\n" + "=" * 50)
print("TEST 3: DataLoader batching")
print("=" * 50)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

batch = next(iter(loader))
print(f"Batch data shape: {batch['data'].shape}")  # Should be (4, 100, 22, 4)
print(f"Batch mask shape: {batch['mask'].shape}")  # Should be (4, 100, 22)
print(f"Number of play IDs: {len(batch['play_id'])}")

print("\n✓ Dataset works!")
