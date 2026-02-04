# NFL Spatiotemporal Representation Learning

Self-supervised learning of play embeddings from NFL Next Gen Stats player tracking data.

## Overview

This project learns compact vector embeddings for NFL plays based purely on player motion over time, using **no labels**. The learned embeddings can be used for:
- Play similarity search
- Clustering similar plays
- Anomaly detection
- Transfer learning for downstream tasks

## Architecture

```
Input: Play tensor (T×N×F)
  ↓
Per-player MLP: Encode each player's features
  ↓
Player pooling: Aggregate across players
  ↓
LSTM: Capture temporal dynamics
  ↓
Output: Play embedding (fixed-size vector)
```

**Key components:**
- **T** = time steps (frames)
- **N** = number of players
- **F** = features per player (x, y, speed, direction)

**Training:** Self-supervised reconstruction (autoencoder)
- Encode play → embedding → decode back to positions
- Loss = MSE on valid (non-padded) positions

## Project Structure

```
nfl_spatiotemporal/
├── data/
│   └── tracking_gameId_2017090700.csv    # NFL tracking data
├── src/
│   ├── preprocessing.py                   # CSV → tensors
│   ├── dataset.py                         # PyTorch Dataset
│   ├── model.py                           # Encoder architecture
│   ├── train.py                           # Training loop
│   └── inference.py                       # Extract embeddings
├── main.py                                # Main entry point
├── requirements.txt                       # Dependencies
└── README.md                              # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn

## Quick Start

### 1. Train the model

```bash
python main.py --data data/tracking_gameId_2017090700.csv --epochs 20
```

**Options:**
- `--data`: Path to tracking CSV
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--output_dir`: Where to save models (default: models/)

### 2. Extract embeddings

```bash
python src/inference.py models/best_model.pt data/tracking_gameId_2017090700.csv
```

This will:
- Load the trained model
- Extract embeddings for all plays
- Find similar plays
- Save embeddings to `embeddings.npz`

### 3. Use embeddings

```python
import numpy as np

# Load embeddings
data = np.load('embeddings.npz', allow_pickle=True)
embeddings = data['embeddings']  # (N, 128) array
play_ids = data['play_ids']      # List of (gameId, playId)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)

# Cluster plays
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit_predict(embeddings)
```

## Data Format

**Input:** NFL Big Data Bowl tracking CSV with columns:
- `gameId`: Game identifier
- `playId`: Play identifier
- `frame.id`: Time step (frame number)
- `nflId`: Player identifier
- `x`, `y`: Player position (yards)
- `s`: Speed (yards/second)
- `dir`: Direction (degrees)

**Output:** Tensor of shape `(T, N, F)` per play:
- T = 100 (max time steps, padded if shorter)
- N = 22 (max players, padded if fewer)
- F = 4 (x, y, speed, direction)

## How It Works

### Preprocessing ([src/preprocessing.py](src/preprocessing.py))

1. **Load CSV**: Group rows by `(gameId, playId)`
2. **Normalize**: (Optional) Flip coordinates so offense always moves left→right
3. **Tensorize**: Convert each play to a `(T, N, F)` tensor
4. **Mask**: Create binary mask for valid (non-padded) positions

**Key function:**
```python
def play_to_tensor(play_df, max_frames=100, max_players=22, features=['x', 'y', 's', 'dir']):
    """Convert DataFrame → (T, N, F) tensor + mask."""
```

### Dataset ([src/dataset.py](src/dataset.py))

PyTorch Dataset that wraps preprocessing:
```python
dataset = NFLPlayDataset('data/tracking.csv')
play = dataset[0]  # Returns {'data': tensor, 'mask': mask, 'play_id': id}
```

Supports batching via `DataLoader`:
```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Yields batches of shape (B, T, N, F)
```

### Model ([src/model.py](src/model.py))

**SpatiotemporalEncoder:**
1. **Per-player MLP**: Process each player independently
   - Input: `(B, T, N, F)` → Output: `(B, T, N, hidden_dim)`
2. **Player pooling**: Mean pooling across players (masked)
   - Output: `(B, T, hidden_dim)`
3. **LSTM**: Capture temporal patterns
   - Output: `(B, T, lstm_hidden_dim)`
4. **Temporal pooling**: Mean pooling over time (masked)
   - Output: `(B, embedding_dim)`

**PlayAutoencoder:**
- Encoder: Play → embedding
- Decoder: Embedding → reconstructed play
- Training objective: Minimize reconstruction error

### Training ([src/train.py](src/train.py))

**Loss function:**
```python
def masked_mse_loss(pred, target, mask):
    """MSE only on valid (non-padded) positions."""
```

**Training loop:**
- 80/20 train/val split
- Adam optimizer
- Save best model based on validation loss
- Early stopping available

## Design Decisions

### Why this architecture?

1. **Per-player MLP**: Each player's motion is encoded independently
   - Allows variable number of players
   - Position-invariant (doesn't depend on player order)

2. **Mean pooling over players**: Simple aggregation
   - Alternative: Attention mechanism (more complex)
   - Works well for baseline

3. **LSTM**: Natural choice for sequences
   - Captures temporal dependencies
   - Alternative: Transformer (future work)

4. **Mean pooling over time**: Aggregates full play
   - Alternative: Use final timestep only
   - Mean pooling more robust to variable play lengths

### Why autoencoder for self-supervision?

**Pros:**
- Simple to implement
- No negative sampling needed (unlike contrastive learning)
- Forces model to capture essential motion patterns

**Alternatives:**
- Contrastive learning (SimCLR-style)
- Predict future frames
- Masked modeling

### Simplicity choices

- **No data augmentation**: Could add noise, temporal masking, etc.
- **No position encoding**: Could help LSTM with absolute time
- **No team/role info**: Could encode offense vs defense
- **Fixed architecture**: Could tune hyperparameters

These are intentional simplifications for a clean baseline!

## Evaluation

**Quantitative:**
- Reconstruction loss (lower = better)
- Downstream task performance (if labels available)

**Qualitative:**
- Similar plays should have similar embeddings
- Clusters should correspond to play types

**Example analysis:**
```python
# Find plays similar to a run play
run_play_idx = 42
similar = find_similar_plays(embeddings, play_ids, run_play_idx, top_k=10)
# Should return other run plays
```

## Future Work

- [ ] Add contrastive learning objective
- [ ] Incorporate team/role information
- [ ] Try Transformer instead of LSTM
- [ ] Add data augmentation
- [ ] Multi-game training
- [ ] Visualization tools
- [ ] Pretrained model weights

## Citation

If you use this code for research, please cite:

```
@misc{nfl-spatiotemporal,
  author = {Your Name},
  title = {Self-Supervised Spatiotemporal Representation Learning for NFL Plays},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/nfl-spatiotemporal}
}
```

## License

MIT License

## Acknowledgments

- Data: [NFL Big Data Bowl](https://www.kaggle.com/c/nfl-big-data-bowl-2021)
- Inspiration: Spatiotemporal representation learning literature
