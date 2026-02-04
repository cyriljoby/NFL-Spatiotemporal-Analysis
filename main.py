"""
Main entry point for the NFL spatiotemporal representation learning pipeline.

Quick start:
    python main.py --data data/tracking_gameId_2017090700.csv --epochs 10
"""

import sys
sys.path.append('src')

from train import train

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train spatiotemporal play embeddings on NFL tracking data'
    )
    parser.add_argument(
        '--data',
        default='data/tracking_gameId_2017090700.csv',
        help='Path to tracking CSV file'
    )
    parser.add_argument(
        '--output_dir',
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='Dimension of play embeddings'
    )

    args = parser.parse_args()

    print("="*60)
    print("NFL Spatiotemporal Representation Learning")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print()

    # Train the model
    model = train(
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim
    )

    print("\n" + "="*60)
    print("Training complete! Next steps:")
    print("="*60)
    print(f"1. Extract embeddings:")
    print(f"   python src/inference.py {args.output_dir}/best_model.pt {args.data}")
    print(f"\n2. Use embeddings for downstream tasks (similarity, clustering, etc.)")
    print("="*60)
