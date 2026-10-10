
import numpy as np
import pytest

from geosentinel.embeddings.vector_store import QdrantVectorStore


@pytest.fixture
def store(tmp_path):
    vector_store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="test_tiles",
        vector_size=512,
    )

    yield vector_store
    vector_store.close()


def test_store_starts_empty(store):
    assert store.count() == 0


def test_add_embedding_increases_count(store):
    embedding = np.ones(512, dtype=np.float32)

    store.add_embedding(
        1,
        embedding,
        {"tile": "tile_001.tif"},
    )

    assert store.count() == 1


def test_search_returns_matching_tile(store):
    embedding = np.random.default_rng(42).random(
        512,
        dtype=np.float32,
    )

    store.add_embedding(
        1,
        embedding,
        {"tile": "tile_001.tif"},
    )

    results = store.search(embedding, limit=1)

    assert len(results) == 1
    assert results[0].payload["tile"] == "tile_001.tif"
    assert results[0].score == pytest.approx(1.0)


def test_search_rejects_wrong_dimension(store):
    embedding = np.zeros(128, dtype=np.float32)

    with pytest.raises(ValueError, match="Expected embedding dimension"):
        store.search(embedding)


def test_add_rejects_wrong_dimension(store):
    embedding = np.zeros(128, dtype=np.float32)

    with pytest.raises(ValueError, match="Expected embedding dimension"):
        store.add_embedding(1, embedding, {})


def test_search_rejects_invalid_limit(store):
    embedding = np.ones(512, dtype=np.float32)

    with pytest.raises(ValueError, match="limit"):
        store.search(embedding, limit=0)
