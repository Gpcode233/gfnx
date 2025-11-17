import chex
import jax
import jax.numpy as jnp
import numpy as np
from Levenshtein import distance as levenshtein_np

##### Distribution distances


def total_variation_distance(p: chex.Array, q: chex.Array) -> chex.Array:
    """
    Compute the Total Variation distance between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).

    Returns:
        Total Variation distance as a scalar.
    """
    chex.assert_equal_shape([p, q])
    return jnp.sum(jnp.abs(p - q)) / 2.0


def kl_divergence(p: chex.Array, q: chex.Array, epsilon: float = 1e-9) -> chex.Array:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).
        epsilon: Small value to avoid division by zero.

    Returns:
        Kullback-Leibler divergence as a scalar.
    """
    chex.assert_equal_shape([p, q])
    return jnp.sum(p * jnp.log(p / (q + epsilon)))


##### String distances


def hamming_distance(s1: chex.Array, s2: chex.Array) -> chex.Array:
    """
    Compute the Hamming distance between two arrays.

    Args:
        s1: First array.
        s2: Second array.

    Returns:
        Hamming distance as a scalar.
    """
    chex.assert_equal_shape([s1, s2])
    return jnp.sum(s1 != s2, dtype=jnp.float32)


def levenstein_distance(s1: chex.Array, s2: chex.Array, nchar: int) -> chex.Array:
    """
    Compute the Levenshtein distance between two strings represented as arrays.

    Args:
        s1: First string as a 1D array.
        s2: Second string as a 1D array.
        nchar: Number of characters in the alphabet (used for filtering).

    Returns:
        Levenshtein distance as a scalar.
    """
    chex.assert_equal_shape([s1, s2])

    # Placeholder for actual implementation
    def callback(a1, a2, nchar) -> float:
        a1 = np.array(a1)
        a2 = np.array(a2)
        a1 = a1[a1 < nchar]
        a2 = a2[a2 < nchar]
        res = levenshtein_np(a1, a2)
        return jnp.float32(res)

    result = jax.pure_callback(callback, jnp.float32(0), s1, s2, nchar)
    return result
