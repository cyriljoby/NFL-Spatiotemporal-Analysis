"""
Training script for spatiotemporal play embeddings.

Self-supervised objective: Reconstruction (autoencoder)
- Encode play to embedding
- Decode embedding back to play
- Loss = MSE on valid (non-padded) positions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

from dataset import NFLPlayDataset
from model import PlayAutoencoder


def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss only on valid (non-padded) positions.

    Args:
        pred: (B, T, N, F) predictions
        target: (B, T, N, F) ground truth
        mask: (B, T, N) binary mask

    Returns:
        loss: scalar
    """
    # Expand mask to match feature dimension
    mask_expanded = mask.unsqueeze(-1)  # (B, T, N, 1)

    # Compute squared error
    squared_error = (pred - target) ** 2  # (B, T, N, F)

    # Apply mask and sum
    masked_error = squared_error * mask_expanded
    total_error = masked_error.sum()

    # Normalize by number of valid entries
    num_valid = mask_expanded.sum() * pred.shape[-1]  # Total valid features
    loss = total_error / num_valid.clamp(min=1)

    return loss


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        data = batch['data'].to(device)    # (B, T, N, F)
        mask = batch['mask'].to(device)    # (B, T, N)

        # Forward pass
        reconstruction, embedding = model(data, mask)

        # Compute loss
        loss = masked_mse_loss(reconstruction, data, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            data = batch['data'].to(device)
            mask = batch['mask'].to(device)

            reconstruction, embedding = model(data, mask)
            loss = masked_mse_loss(reconstruction, data, mask)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    data_path,
    output_dir='models',
    batch_size=32,
    num_epochs=20,
    learning_rate=1e-3,
    embedding_dim=128,
    device=None
):
    """
    Main training function.

    Args:
        data_path: Path to tracking CSV file
        output_dir: Directory to save checkpoints
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        embedding_dim: Size of play embeddings
        device: 'cuda' or 'cpu' (auto-detect if None)
    """
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading dataset...")
    dataset = NFLPlayDataset(data_path)

    # Simple train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create model
    print("\nInitializing model...")
    model = PlayAutoencoder(
        input_dim=4,
        player_hidden_dim=32,
        lstm_hidden_dim=64,
        embedding_dim=embedding_dim,
        max_frames=100,
        max_players=22
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    final_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_path)

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}/")
    print(f"{'='*50}")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train spatiotemporal play encoder')
    parser.add_argument('data_path', help='Path to tracking CSV file')
    parser.add_argument('--output_dir', default='models', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim
    )
