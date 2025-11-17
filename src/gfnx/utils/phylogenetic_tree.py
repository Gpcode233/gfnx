import json
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp

from ..environment.phylogenetic_tree import (
    CHARACTERS_MAPS,
    CONFIGS,
    PhyloTreeEnvironment,
)
from ..reward.phylogenetic_tree import PhyloTreeRewardModule


def get_phylo_gfn_env(
    dataset_name: str, data_folder: Path, **kwargs
) -> Tuple[PhyloTreeEnvironment, Tuple[str, float, float]]:
    """
    Create phylogenetic tree environment with predefined configurations.

    Args:
        dataset_name: Name of the dataset (DS1-DS8)
        data_folder: Path to folder containing dataset JSON files
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        Tuple of (environment, configuration)
    """
    if dataset_name not in CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    with open(data_folder / f"{dataset_name}.json", "r") as f:
        sequences_dict = json.load(f)

    sequence_type, C, scale, bits_per_seq_elem = CONFIGS[dataset_name]

    char_dict = CHARACTERS_MAPS[sequence_type]
    sequences = jnp.array(
        [
            [char_dict[c] for c in sequence]
            for sequence in sequences_dict.values()
        ],
        dtype=jnp.uint8,
    )

    reward_module = PhyloTreeRewardModule(
        num_nodes=len(sequences_dict), scale=scale, C=C
    )

    env = PhyloTreeEnvironment(
        reward_module=reward_module,
        sequences=sequences,
        sequence_type=sequence_type,
        bits_per_seq_elem=bits_per_seq_elem,
        **kwargs,
    )

    return env, CONFIGS[dataset_name]
