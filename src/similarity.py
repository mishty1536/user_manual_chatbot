"""Cosine similarity search using numpy."""

from typing import List, Tuple
import numpy as np

from .config import TOP_K_CHUNKS


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (-1 to 1)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def find_top_k_similar(
    query_embedding: List[float],
    stored_embeddings: List[List[float]],
    k: int = TOP_K_CHUNKS
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar embeddings.

    Args:
        query_embedding: Query vector
        stored_embeddings: List of stored embedding vectors
        k: Number of top results to return

    Returns:
        List of (index, similarity_score) tuples, sorted by score descending
    """
    if not stored_embeddings:
        return []

    query = np.array(query_embedding)
    stored = np.array(stored_embeddings)

    # Normalize vectors
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return [(i, 0.0) for i in range(min(k, len(stored_embeddings)))]

    query_normalized = query / query_norm

    # Normalize stored embeddings
    stored_norms = np.linalg.norm(stored, axis=1, keepdims=True)
    # Avoid division by zero
    stored_norms = np.where(stored_norms == 0, 1, stored_norms)
    stored_normalized = stored / stored_norms

    # Compute similarities
    similarities = np.dot(stored_normalized, query_normalized)

    # Get top-k indices
    k = min(k, len(similarities))
    top_indices = np.argsort(similarities)[-k:][::-1]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]
