"""Non-executable checkpoints for exact optimizer and random-stream continuation."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np


def save_checkpoint(path: Path, state, metadata: dict) -> None:
    leaves = jax.tree_util.tree_leaves(state)
    arrays = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_checkpoint(path: Path, template):
    expected, tree = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *(f"leaf_{i}" for i in range(len(expected)))}:
            raise ValueError("Checkpoint state does not match model/optimizer structure")
        leaves = [archive[f"leaf_{i}"] for i in range(len(expected))]
        if any(a.shape != b.shape or a.dtype != b.dtype
               for a, b in zip(leaves, expected, strict=True)):
            raise ValueError("Checkpoint leaf shape/dtype mismatch")
        metadata = json.loads(str(archive["metadata"]))
    return jax.tree_util.tree_unflatten(tree, leaves), metadata
