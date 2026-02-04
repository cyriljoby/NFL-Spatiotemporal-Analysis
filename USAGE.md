# Usage Guide

## Step-by-Step Tutorial

### Step 1: Understand the Data

Your tracking CSV has these columns:
- `gameId`, `playId`: Identify each play
- `frame.id`: Time step (1, 2, 3, ...)
- `nflId`: Player ID
- `x`, `y`: Position on field (yards)
- `s`: Speed (yards/second)
- `dir`: Direction (degrees)

Each play has ~50-100 frames, 22 players per frame.

### Step 2: Train the Model

```bash
# Quick test (1 epoch)
python main.py --data data/tracking_gameId_2017090700.csv --epochs 1

# Full training (20 epochs)
python main.py --data data/tracking_gameId_2017090700.csv --epochs 20

# Custom settings
python main.py \
  --data data/tracking_gameId_2017090700.csv \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.0005 \
  --embedding_dim 256
```

**What happens:**
1. Loads all plays from CSV
2. Converts each to (100, 22, 4) tensor
3. Trains autoencoder to reconstruct plays
4. Saves best model to `models/best_model.pt`

**Training output:**
```
Epoch 1/20
Train loss: 12282.16
Val loss: 11534.28
✓ Saved best model
```

Lower loss = better reconstruction = better embeddings.

### Step 3: Extract Embeddings

```bash
python src/inference.py models/best_model.pt data/tracking_gameId_2017090700.csv
```

**Output:**
- `embeddings.npz`: Contains embeddings and play IDs
- Console: Shows similar plays to play 0

### Step 4: Use Embeddings

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
data = np.load('embeddings.npz', allow_pickle=True)
embeddings = data['embeddings']  # (177, 128)
play_ids = data['play_ids']

# Example 1: Find similar plays
query_idx = 0
similarities = cosine_similarity(embeddings[query_idx:query_idx+1], embeddings)[0]
top_5 = np.argsort(similarities)[::-1][1:6]  # Exclude self
print(f"Plays similar to {play_ids[query_idx]}:")
for idx in top_5:
    print(f"  {play_ids[idx]} (similarity: {similarities[idx]:.3f})")

# Example 2: Cluster plays
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

for i in range(10):
    cluster_plays = [play_ids[j] for j in range(len(play_ids)) if clusters[j] == i]
    print(f"Cluster {i}: {len(cluster_plays)} plays")

# Example 3: Visualize with t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.colorbar(label='Cluster')
plt.title('Play Embeddings (t-SNE)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('play_embeddings.png')
plt.show()
```

## Understanding the Output

### What are embeddings?

Each play becomes a 128-dimensional vector (by default):
```
Play 1: [0.23, -0.45, 0.12, ..., 0.89]  # 128 numbers
Play 2: [-0.11, 0.56, -0.34, ..., 0.23]
```

**Similar plays** → **similar vectors** (high cosine similarity)

### How to interpret

- **High similarity (>0.9)**: Plays have very similar motion patterns
- **Medium similarity (0.7-0.9)**: Some common patterns
- **Low similarity (<0.7)**: Different types of plays

### What makes plays similar?

The model learns patterns like:
- Overall direction of movement
- Speed profiles
- Player formations and routes
- Temporal dynamics

**Without any labels!** It's purely based on motion.

## Advanced Usage

### Training on Multiple Games

```python
import pandas as pd
import sys
sys.path.append('src')
from train import train

# Combine multiple game files
game_files = [
    'data/tracking_gameId_2017090700.csv',
    'data/tracking_gameId_2017091000.csv',
    # ... more files
]

dfs = [pd.read_csv(f) for f in game_files]
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv('data/all_games.csv', index=False)

# Train on combined data
train('data/all_games.csv', num_epochs=50, batch_size=64)
```

### Custom Architecture

Edit `src/model.py`:
```python
model = SpatiotemporalEncoder(
    input_dim=4,
    player_hidden_dim=64,    # Increase capacity
    lstm_hidden_dim=128,     # Larger LSTM
    embedding_dim=256,       # Bigger embeddings
    num_lstm_layers=2        # Stack LSTMs
)
```

### Resume Training

```python
import torch
from src.model import PlayAutoencoder
from src.train import train

# Load checkpoint
checkpoint = torch.load('models/best_model.pt')
model = PlayAutoencoder(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# ... continue training loop
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python main.py --batch_size 8

# Or reduce model size
# Edit src/model.py to use smaller hidden dimensions
```

### Training too slow

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Use smaller max_frames/max_players in dataset.py
```

### Poor embeddings

- Train longer (more epochs)
- Use more data (multiple games)
- Tune hyperparameters (learning rate, architecture)
- Try different self-supervised objectives

## Next Steps

1. **Evaluate qualitatively**: Do similar plays actually look similar?
2. **Downstream tasks**: Use embeddings for play classification
3. **Visualization**: Plot embeddings, cluster analysis
4. **Improve**: Try different architectures, objectives

Good luck with your research!
