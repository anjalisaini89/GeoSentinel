import numpy as np
import pytest

from geosentinel.embeddings.base import (
    DummyEmbedder,
    ImageEmbedder,
)


def test_image_embedder_is_abstract():
    assert ImageEmbedder.__abstractmethods__


def test_dummy_embedder():
    embedder = DummyEmbedder(
        embedding_dimension=128
    )

    image = np.random.rand(
        3,
        128,
        128,
    ).astype(np.float32)

    embedding = embedder.embed(image)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (128,)
    assert embedding.dtype == np.float32


def test_embedding_is_normalized():
    embedder = DummyEmbedder(
        embedding_dimension=128
    )

    image = np.random.rand(
        3,
        128,
        128,
    ).astype(np.float32)

    embedding = embedder.embed(image)

    assert np.isclose(
        np.linalg.norm(embedding),
        1.0,
        atol=1e-6,
    )


def test_embedding_is_deterministic():
    embedder = DummyEmbedder(
        embedding_dimension=64
    )

    image = np.ones(
        (3, 128, 128),
        dtype=np.float32,
    )

    embedding_1 = embedder.embed(image)
    embedding_2 = embedder.embed(image)

    assert np.array_equal(
        embedding_1,
        embedding_2,
    )


def test_invalid_image_dimensions():
    embedder = DummyEmbedder()

    image = np.random.rand(
        128,
        128,
    ).astype(np.float32)

    with pytest.raises(
        ValueError,
        match="CHW format",
    ):
        embedder.embed(image)


def test_empty_image():
    embedder = DummyEmbedder()

    image = np.empty(
        (3, 0, 0),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        embedder.embed(image)


def test_invalid_embedding_dimension():
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        DummyEmbedder(
            embedding_dimension=0
        )