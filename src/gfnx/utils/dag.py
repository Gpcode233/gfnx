from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp


def load_dag_samples(samples_path: Path) -> Dict[str, Any]:
    """
    Loads samples and extracts basic information needed for DAG environment setup.

    Args:
        samples_path: Path to the file containing the samples.

    Returns:
        Dictionary with initialization arguments:
        - samples: The loaded samples array
        - num_variables: Number of variables in the samples
    """
    samples = jnp.load(samples_path)
    num_variables = samples.shape[1]

    return {"samples": samples, "num_variables": num_variables}
