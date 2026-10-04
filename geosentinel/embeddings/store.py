from pathlib import Path
from typing import Any
import json

import numpy as np


def save_embedding(
    embedding: np.ndarray,
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save an embedding and its metadata to a compressed NumPy file.
    """

    output_path = Path(output_path)

    if embedding.ndim != 1:
        raise ValueError("Embedding must be a 1D vector.")

    if embedding.size == 0:
        raise ValueError("Embedding cannot be empty.")

    if not metadata:
        raise ValueError("Metadata cannot be empty.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_json = json.dumps(metadata)

    np.savez_compressed(
        output_path,
        embedding=embedding.astype(np.float32),
        metadata=metadata_json,
    )

    return output_path


def load_embedding(
    file_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load an embedding and its metadata.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {file_path}"
        )

    with np.load(
        file_path,
        allow_pickle=False,
    ) as data:
        if "embedding" not in data:
            raise ValueError(
                "Embedding file does not contain an embedding."
            )

        if "metadata" not in data:
            raise ValueError(
                "Embedding file does not contain metadata."
            )

        embedding = data["embedding"].astype(np.float32)

        metadata = json.loads(
            str(data["metadata"])
        )

    if embedding.ndim != 1:
        raise ValueError(
            "Stored embedding must be a 1D vector."
        )

    return embedding, metadata