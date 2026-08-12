"""
Math utilities for Famou 2.0.

Shared mathematical functions for vector operations, distance metrics, etc.
"""

import math
from typing import List, Optional

import numpy as np


def cosine_distance(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance = 1 - cosine_similarity

    Args:
        vec1: First vector (list of floats)
        vec2: Second vector (list of floats)

    Returns:
        Cosine distance in range [0, 2], where 0 means identical direction,
        1 means orthogonal, 2 means opposite direction.
        Returns 1.0 if either vector is None or empty.

    Example:
        >>> cosine_distance([1, 0], [0, 1])  # Orthogonal
        1.0
        >>> cosine_distance([1, 0], [1, 0])  # Same direction
        0.0
    """
    if not vec1 or not vec2:
        return 1.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    return 1.0 - (dot / (norm1 * norm2))


def cosine_similarity(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector (list of floats)
        vec2: Second vector (list of floats)

    Returns:
        Cosine similarity in range [-1, 1], where 1 means identical direction,
        0 means orthogonal, -1 means opposite direction.
        Returns 0.0 if either vector is None or empty.

    Example:
        >>> cosine_similarity([1, 0], [1, 0])
        1.0
        >>> cosine_similarity([1, 0], [0, 1])
        0.0
    """
    if not vec1 or not vec2:
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def normalize_vector(vec: Optional[List[float]]) -> List[float]:
    """
    Normalize vector to unit length.

    Args:
        vec: Input vector

    Returns:
        Normalized vector with unit length, or original if empty/zero-norm

    Example:
        >>> normalize_vector([3, 4])
        [0.6, 0.8]
    """
    if not vec:
        return vec or []

    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec

    return [x / norm for x in vec]


def compute_mean_vector(vectors: List[List[float]]) -> List[float]:
    """
    Compute element-wise mean of vectors.

    Args:
        vectors: List of vectors (all should have same dimension)

    Returns:
        Mean vector, or empty list if no vectors provided

    Example:
        >>> compute_mean_vector([[1, 2], [3, 4]])
        [2.0, 3.0]
    """
    if not vectors:
        return []

    dim = max(len(v) for v in vectors)
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    return [x / len(vectors) for x in result]


def compute_sum_vector(vectors: List[List[float]]) -> List[float]:
    """
    Compute element-wise sum of vectors.

    Args:
        vectors: List of vectors (all should have same dimension)

    Returns:
        Sum vector, or empty list if no vectors provided

    Example:
        >>> compute_sum_vector([[1, 2], [3, 4]])
        [4, 6]
    """
    if not vectors:
        return []

    dim = max(len(v) for v in vectors)
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    return result


def stable_softmax(x: List[float], temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax with temperature scaling.

    Args:
        x: Input scores/logits
        temperature: Temperature parameter (higher = more uniform, lower = more peaked)

    Returns:
        Probability distribution (numpy array)

    Example:
        >>> stable_softmax([1, 2, 3], temperature=1.0)
        array([0.09, 0.24, 0.67])  # approximately
    """
    x = np.array(x)
    x = x / max(temperature, 1e-8)
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    return x_exp / x_exp.sum()


def euclidean_distance(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance, or infinity if either vector is None/empty

    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    if not vec1 or not vec2:
        return float("inf")

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def vectors_equal(vec1: List[float], vec2: List[float], tolerance: float = 1e-9) -> bool:
    """
    Check if two vectors are approximately equal.

    Args:
        vec1: First vector
        vec2: Second vector
        tolerance: Maximum allowed difference per element

    Returns:
        True if vectors are equal within tolerance

    Example:
        >>> vectors_equal([1.0, 2.0], [1.0, 2.0])
        True
        >>> vectors_equal([1.0, 2.0], [1.0, 2.1])
        False
    """
    if len(vec1) != len(vec2):
        return False
    return all(abs(a - b) < tolerance for a, b in zip(vec1, vec2))
