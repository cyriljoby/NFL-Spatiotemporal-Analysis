# Learning Guide: Understanding the Complete Pipeline

This guide walks you through the entire codebase to help you learn how it all works.

## Overview: What Did We Build?

A complete machine learning pipeline that:
1. **Loads** NFL tracking data (CSV)
2. **Converts** to tensors (3D arrays)
3. **Trains** a neural network to learn play embeddings
4. **Extracts** embeddings for similarity search and analysis

**Goal**: Turn raw player positions → compact vector representations

## Learning Path

### Level 1: Data Processing (Start Here!)

#### File: [src/preprocessing.py](src/preprocessing.py)

**Core concept**: Convert tabular data → tensors

**Key functions to understand:**

1. **`load_and_group_plays()`**
   - What it does: Groups CSV rows by play
   - Input: CSV path
   - Output: Dict mapping (gameId, playId) → DataFrame
   - Learn: Pandas groupby operations

2. **`play_to_tensor()`** ⭐ Most important!
   - What it does: Converts one play DataFrame → (T, N, F) tensor
   - Input: DataFrame with frame.id, nflId, x, y, s, dir
   - Output:
     - `data`: (100, 22, 4) tensor - player positions over time
     - `mask`: (100, 22) tensor - which positions are valid (not padding)

   **How it works:**
   ```
   1. Get unique frames (1, 2, 3, ..., 98)
   2. Get unique players (22 player IDs)
   3. Create empty tensor filled with zeros
   4. Loop over frames:
      For each player:
        Fill in [x, y, speed, direction]
        Mark mask = 1 (valid data)
   5. Remaining entries stay 0 with mask = 0 (padding)
   ```

**Exercise**: Read the line-by-line explanation I gave earlier. Try to trace through one example play manually.

---

### Level 2: PyTorch Dataset

#### File: [src/dataset.py](src/dataset.py)

**Core concept**: Wrap preprocessing in PyTorch format

**Key class: `NFLPlayDataset`**

**What it does:**
- `__init__`: Load CSV, convert ALL plays to tensors, store in memory
- `__len__`: Return number of plays
- `__getitem__`: Return one play (data, mask, play_id)

**Why PyTorch Dataset?**
- Standard interface for training
- Works with `DataLoader` for batching
- Handles shuffling, parallel loading

**How batching works:**
```python
dataset[0]  # Returns (100, 22, 4) for play 0
dataset[1]  # Returns (100, 22, 4) for play 1

# DataLoader stacks them:
batch = stack([dataset[0], dataset[1], dataset[2], dataset[3]])
# Result: (4, 100, 22, 4) - 4 plays at once!
```

**Exercise**:
- Run [test_dataset.py](test_dataset.py)
- Look at batch shapes
- Try changing batch_size

---

### Level 3: Neural Network Architecture

#### File: [src/model.py](src/model.py)

**Core concept**: Encode spatiotemporal data → fixed-size embedding

**Two models:**

1. **`SpatiotemporalEncoder`** ⭐ Core architecture
   ```
   Input: (B, T, N, F) = (batch, time, players, features)

   Step 1: Per-player MLP
     Process each player independently
     (B, T, N, F) → (B, T, N, hidden_dim)

   Step 2: Player pooling
     Average across players (masked mean)
     (B, T, N, hidden_dim) → (B, T, hidden_dim)

   Step 3: LSTM
     Capture temporal patterns
     (B, T, hidden_dim) → (B, T, lstm_hidden)

   Step 4: Temporal pooling
     Average across time (masked mean)
     (B, T, lstm_hidden) → (B, lstm_hidden)

   Step 5: Final projection
     (B, lstm_hidden) → (B, embedding_dim)
   ```

2. **`PlayAutoencoder`**
   - Encoder: Use SpatiotemporalEncoder
   - Decoder: MLP to reconstruct play
   - Training: Minimize reconstruction error

**Key concepts to understand:**

- **Per-player MLP**: Why? Each player processed independently, regardless of which "slot" they're in
- **Masked pooling**: Why? Some plays have <22 players, <100 frames. Only average valid positions.
- **LSTM**: Why? Captures temporal dependencies (earlier frames influence later)
- **Autoencoder**: Why? Self-supervised - no labels needed. Forces model to learn meaningful patterns.

**Exercise**:
- Trace a tensor through each step
- Start with (4, 100, 22, 4)
- Calculate shape after each operation
- Understand reshaping operations

---

### Level 4: Training Loop

#### File: [src/train.py](src/train.py)

**Core concept**: Optimize model to minimize reconstruction error

**Key functions:**

1. **`masked_mse_loss()`**
   ```python
   # Don't compute loss on padding!
   error = (pred - target)^2
   masked_error = error * mask  # Zero out padding
   loss = mean(masked_error)
   ```

2. **`train_epoch()`**
   ```python
   For each batch:
     1. Forward pass: data → reconstruction
     2. Compute loss
     3. Backward pass: compute gradients
     4. Update weights
   ```

3. **`validate()`**
   - Same as train but no weight updates
   - Check if model is overfitting

4. **`train()`** - Main training loop
   - Load data
   - Create model
   - Train for N epochs
   - Save best model

**Training process:**
```
Epoch 1:
  - Feed batch through model
  - Compare reconstruction to original
  - Update weights to reduce error
  - Repeat for all batches
  - Validate on held-out data

Epoch 2:
  - Same process, model gets better
  ...

Epoch 20:
  - Model has learned good embeddings
```

**Exercise**:
- Run training with `--epochs 1`
- Watch loss decrease during training
- Compare train vs validation loss

---

### Level 5: Inference & Usage

#### File: [src/inference.py](src/inference.py)

**Core concept**: Use trained model to extract embeddings

**Key functions:**

1. **`load_model()`**
   - Load saved checkpoint
   - Initialize model with same architecture
   - Load learned weights

2. **`extract_embeddings()`**
   ```python
   For each play:
     1. Convert to tensor
     2. Pass through encoder
     3. Get embedding vector
     4. Store in array

   Result: (N, 128) array where each row is a play embedding
   ```

3. **`find_similar_plays()`**
   - Compute cosine similarity between embeddings
   - High similarity = similar motion patterns
   - Find nearest neighbors

**Why it works:**
- Model learned to compress plays into embeddings
- Similar plays get similar embeddings
- Can now search, cluster, analyze without labels!

**Exercise**:
- Extract embeddings for your data
- Find similar plays manually
- Do they actually look similar?

---

## Key Concepts Explained

### 1. Tensors vs DataFrames

**DataFrame (pandas):**
```
frame.id | nflId | x    | y    | s   | dir
---------|-------|------|------|-----|-----
1        | 123   | 10.5 | 25.0 | 5.2 | 90
1        | 456   | 15.0 | 30.0 | 4.1 | 85
2        | 123   | 11.0 | 25.2 | 5.4 | 92
```

**Tensor (PyTorch):**
```python
data[0, 0, :] = [10.5, 25.0, 5.2, 90]  # Frame 0, Player 0
data[0, 1, :] = [15.0, 30.0, 4.1, 85]  # Frame 0, Player 1
data[1, 0, :] = [11.0, 25.2, 5.4, 92]  # Frame 1, Player 0
```

**Why tensors?**
- Neural networks need fixed-size numeric arrays
- GPUs are optimized for tensor operations
- Batching requires consistent shapes

### 2. Masking

**Problem:** Not all plays have 22 players or 100 frames

**Solution:** Pad with zeros, track with mask
```python
data[98, 0, :] = [0, 0, 0, 0]  # Real data
mask[98, 0] = 1

data[99, 0, :] = [0, 0, 0, 0]  # Padding
mask[99, 0] = 0  # Ignore this!
```

**Usage:**
- During pooling: Only average valid positions
- During loss: Only compute loss on valid positions

### 3. Self-Supervised Learning

**Traditional supervised:**
```
Input: Play data
Label: "Run" or "Pass"
Train: Predict label
```

**Self-supervised (our approach):**
```
Input: Play data
Target: Play data itself
Train: Reconstruct input
```

**Why?**
- No labels needed!
- Model learns structure of data
- Embeddings capture motion patterns

### 4. Embeddings

**What are they?**
- Fixed-size vector representation of variable-size input
- Similar inputs → similar vectors

**How are they learned?**
- Encoder compresses play → small vector
- Decoder expands vector → reconstructed play
- Training forces encoder to keep important info

**Why useful?**
- Play similarity (cosine similarity of embeddings)
- Clustering (k-means on embeddings)
- Downstream tasks (use as features)

---

## Exercises to Deepen Understanding

### Beginner

1. **Trace data flow:**
   - Pick one play from CSV
   - Manually trace how it becomes a tensor
   - Check your work against the code

2. **Understand shapes:**
   - Print tensor shapes at each model step
   - Verify they match documentation

3. **Modify hyperparameters:**
   - Change embedding_dim from 128 to 64
   - Change batch_size from 32 to 16
   - Observe impact on training

### Intermediate

4. **Implement new feature:**
   - Add "acceleration" to features (x, y, s, dir, accel)
   - Modify preprocessing and model

5. **Try different architecture:**
   - Replace mean pooling with max pooling
   - Add more LSTM layers
   - Try GRU instead of LSTM

6. **Add data augmentation:**
   - Random temporal masking
   - Add noise to positions
   - Flip plays horizontally

### Advanced

7. **Implement contrastive learning:**
   - Instead of reconstruction loss
   - Positive pairs: Similar plays
   - Negative pairs: Different plays
   - Use SimCLR-style loss

8. **Add attention mechanism:**
   - Replace mean pooling with attention
   - Learn which players/timesteps are important

9. **Visualization:**
   - t-SNE of embeddings
   - Animate plays
   - Show nearest neighbors

---

## Common Questions

### Q: Why LSTM instead of Transformer?

**A:** Simplicity for baseline. Transformers are more powerful but more complex. LSTM is easier to understand and works well for sequences.

### Q: Why autoencoder instead of contrastive learning?

**A:** Simpler to implement. No need for negative sampling or augmentations. Good baseline before trying more complex methods.

### Q: How do I know if embeddings are good?

**A:** Qualitative check: Do similar plays have high similarity? Quantitative: Use for downstream task (if you have labels).

### Q: Can I use this for other sports?

**A:** Yes! Just need tracking data in similar format. Basketball, soccer, etc. would work with minor modifications.

### Q: How much data do I need?

**A:** More is better. One game (177 plays) is enough to test, but multiple games will give better embeddings.

---

## Next Steps

1. **Read the code in this order:**
   - [preprocessing.py](src/preprocessing.py)
   - [dataset.py](src/dataset.py)
   - [model.py](src/model.py)
   - [train.py](src/train.py)
   - [inference.py](src/inference.py)

2. **Run the tests:**
   - `python test_data_exploration.py`
   - `python test_dataset.py`

3. **Train a model:**
   - `python main.py --epochs 10`

4. **Extract and analyze embeddings:**
   - `python src/inference.py models/best_model.pt data/tracking.csv`
   - Load embeddings.npz in a notebook
   - Do similarity search, clustering, visualization

5. **Experiment:**
   - Modify architecture
   - Try different self-supervised objectives
   - Add your own ideas!

---

## Resources for Learning More

**Spatiotemporal ML:**
- "Attention is All You Need" (Transformers)
- "A Simple Framework for Contrastive Learning" (SimCLR)
- "Masked Autoencoders Are Scalable Vision Learners" (MAE)

**PyTorch:**
- Official tutorials: pytorch.org/tutorials
- "Deep Learning with PyTorch" (book)

**Sports Analytics:**
- NFL Big Data Bowl winners
- Tracking data papers from various sports

Good luck with your research! Feel free to modify and extend this codebase.
