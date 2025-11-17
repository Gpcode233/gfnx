import chex
import jax.numpy as jnp


def total_variation_distance(p: chex.Array, q: chex.Array) -> chex.Array:
    """
    Compute the Total Variation distance between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).

    Returns:
        Total Variation distance as a scalar.
    """
    return jnp.sum(jnp.abs(p - q)) / 2.0


def kl_divergence(
    p: chex.Array, q: chex.Array, epsilon: float = 1e-9
) -> chex.Array:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).
        epsilon: Small value to avoid division by zero.

    Returns:
        Kullback-Leibler divergence as a scalar.
    """  # noqa: E501
    return jnp.sum(p * jnp.log(p / (q + epsilon)))
