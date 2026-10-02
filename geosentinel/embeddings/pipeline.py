from pathlib import Path

import numpy as np

from geosentinel.embeddings.base import ImageEmbedder
from geosentinel.preprocessing.raster import preprocess_raster


def generate_embedding(
    file_path: str | Path,
    embedder: ImageEmbedder,
    expected_bands: int = 3,
) -> np.ndarray:
    """
    Preprocess a raster and generate its embedding.

    Parameters
    ----------
    file_path:
        Path to the raster tile.
    embedder:
        An implementation of ImageEmbedder.
    expected_bands:
        Number of expected raster bands.

    Returns
    -------
    np.ndarray
        One-dimensional embedding vector.
    """

    image = preprocess_raster(
        file_path,
        expected_bands=expected_bands,
    )

    embedding = embedder.embed(image)

    if embedding.ndim != 1:
        raise ValueError(
            "Embedding model must return a 1D vector."
        )

    if embedding.size == 0:
        raise ValueError(
            "Embedding vector cannot be empty."
        )

    return embedding.astype(np.float32)