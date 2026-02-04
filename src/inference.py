"""
Inference script: Extract embeddings from trained model.

Shows how to:
1. Load a trained model
2. Get embeddings for plays
3. Compute play similarity
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dataset import NFLPlayDataset
from model import PlayAutoencoder


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    model = PlayAutoencoder(
        input_dim=4,
        player_hidden_dim=32,
        lstm_hidden_dim=64,
        embedding_dim=128,
        max_frames=100,
        max_players=22
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs, val_loss: {checkpoint['val_loss']:.4f}")

    return model


def extract_embeddings(model, dataset, device='cpu'):
    """
    Extract embeddings for all plays in dataset.

    Returns:
        embeddings: (N, embedding_dim) array
        play_ids: List of (gameId, playId) tuples
    """
    model.eval()
    embeddings_list = []
    play_ids_list = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            data = sample['data'].unsqueeze(0).to(device)  # Add batch dim
            mask = sample['mask'].unsqueeze(0).to(device)

            # Get embedding
            embedding = model.encode(data, mask)  # (1, embedding_dim)
            embeddings_list.append(embedding.cpu().numpy())
            play_ids_list.append(sample['play_id'])

    embeddings = np.vstack(embeddings_list)  # (N, embedding_dim)
    return embeddings, play_ids_list


def find_similar_plays(embeddings, play_ids, query_idx, top_k=5):
    """
    Find most similar plays to a query play.

    Args:
        embeddings: (N, embedding_dim) array
        play_ids: List of play IDs
        query_idx: Index of query play
        top_k: Number of similar plays to return

    Returns:
        List of (idx, play_id, similarity) tuples
    """
    query_embedding = embeddings[query_idx:query_idx+1]  # (1, embedding_dim)

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]  # (N,)

    # Get top-k (excluding the query itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]

    results = []
    for idx in top_indices:
        results.append((idx, play_ids[idx], similarities[idx]))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract play embeddings')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('data_path', help='Path to tracking CSV')
    parser.add_argument('--query_idx', type=int, default=0, help='Query play index for similarity search')
    parser.add_argument('--top_k', type=int, default=5, help='Number of similar plays to find')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device=device)

    # Load dataset
    print("\nLoading dataset...")
    dataset = NFLPlayDataset(args.data_path)

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings, play_ids = extract_embeddings(model, dataset, device=device)
    print(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # Find similar plays
    print(f"\nFinding plays similar to play {args.query_idx} (ID: {play_ids[args.query_idx]})...")
    similar_plays = find_similar_plays(embeddings, play_ids, args.query_idx, args.top_k)

    print(f"\nTop {args.top_k} most similar plays:")
    for rank, (idx, play_id, similarity) in enumerate(similar_plays, 1):
        print(f"  {rank}. Play {idx} (ID: {play_id}) - Similarity: {similarity:.4f}")

    # Save embeddings
    output_path = 'embeddings.npz'
    np.savez(output_path, embeddings=embeddings, play_ids=play_ids)
    print(f"\n✓ Saved embeddings to {output_path}")


if __name__ == '__main__':
    main()
