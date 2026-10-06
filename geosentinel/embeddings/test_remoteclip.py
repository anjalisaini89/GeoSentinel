from pathlib import Path

import numpy as np
import pytest

from geosentinel.embeddings.remoteclip import RemoteCLIPEmbedder


CHECKPOINT = Path("models/remoteclip/RemoteCLIP-ViT-B-32.pt")


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="RemoteCLIP checkpoint is not available.",
)
def test_remoteclip_embedding_shape():
    embedder = RemoteCLIPEmbedder(CHECKPOINT)

    image = np.random.default_rng(42).random(
        (3, 224, 224),
        dtype=np.float32,
    )

    embedding = embedder.embed(image)

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="RemoteCLIP checkpoint is not available.",
)
def test_remoteclip_embedding_is_normalized():
    embedder = RemoteCLIPEmbedder(CHECKPOINT)

    image = np.random.default_rng(42).random(
        (3, 224, 224),
        dtype=np.float32,
    )

    embedding = embedder.embed(image)

    norm = np.linalg.norm(embedding)

    assert np.isclose(norm, 1.0, atol=1e-5)


def test_remoteclip_rejects_non_three_channel_image():
    if not CHECKPOINT.exists():
        pytest.skip("RemoteCLIP checkpoint is not available.")

    embedder = RemoteCLIPEmbedder(CHECKPOINT)

    image = np.zeros(
        (4, 224, 224),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="exactly 3"):
        embedder.embed(image)