import numpy as np
import pytest

from geosentinel.embeddings.store import (
    load_embedding,
    save_embedding,
)


def test_save_and_load_embedding(tmp_path):
    embedding = np.random.rand(
        128
    ).astype(np.float32)

    metadata = {
        "tile_id": "sample_r0_c0",
        "source": "sample.tif",
        "crs": "EPSG:4326",
        "width": 128,
        "height": 128,
    }

    output_path = tmp_path / "embedding.npz"

    saved_path = save_embedding(
        embedding,
        metadata,
        output_path,
    )

    assert saved_path.exists()

    loaded_embedding, loaded_metadata = load_embedding(
        saved_path
    )

    assert loaded_embedding.shape == (128,)
    assert loaded_embedding.dtype == np.float32

    assert np.array_equal(
        loaded_embedding,
        embedding,
    )

    assert loaded_metadata == metadata


def test_save_rejects_non_1d_embedding(tmp_path):
    embedding = np.random.rand(
        3,
        128,
    ).astype(np.float32)

    metadata = {
        "tile_id": "test",
    }

    with pytest.raises(
        ValueError,
        match="1D vector",
    ):
        save_embedding(
            embedding,
            metadata,
            tmp_path / "embedding.npz",
        )


def test_save_rejects_empty_embedding(tmp_path):
    embedding = np.array(
        [],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        save_embedding(
            embedding,
            {"tile_id": "test"},
            tmp_path / "embedding.npz",
        )


def test_save_rejects_empty_metadata(tmp_path):
    embedding = np.ones(
        128,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Metadata cannot be empty",
    ):
        save_embedding(
            embedding,
            {},
            tmp_path / "embedding.npz",
        )


def test_load_missing_embedding():
    with pytest.raises(FileNotFoundError):
        load_embedding(
            "data/embeddings/does_not_exist.npz"
        )