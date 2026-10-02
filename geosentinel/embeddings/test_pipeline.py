import numpy as np
import pytest

from geosentinel.embeddings.base import DummyEmbedder
from geosentinel.embeddings.pipeline import generate_embedding


def test_generate_embedding():
    embedder = DummyEmbedder(
        embedding_dimension=128
    )

    embedding = generate_embedding(
        "data/samples/sample.tif",
        embedder,
        expected_bands=3,
    )

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (128,)
    assert embedding.dtype == np.float32


def test_generated_embedding_is_normalized():
    embedder = DummyEmbedder(
        embedding_dimension=128
    )

    embedding = generate_embedding(
        "data/samples/sample.tif",
        embedder,
        expected_bands=3,
    )

    assert np.isclose(
        np.linalg.norm(embedding),
        1.0,
        atol=1e-6,
    )


def test_pipeline_rejects_missing_raster():
    embedder = DummyEmbedder()

    with pytest.raises(FileNotFoundError):
        generate_embedding(
            "data/samples/does_not_exist.tif",
            embedder,
        )


def test_pipeline_rejects_wrong_band_count():
    embedder = DummyEmbedder()

    with pytest.raises(
        ValueError,
        match="Expected 4 bands",
    ):
        generate_embedding(
            "data/samples/sample.tif",
            embedder,
            expected_bands=4,
        )