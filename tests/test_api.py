v
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from geosentinel.api.main import app


class FakeEmbedder:
    def embed_text(self, text):
        assert text == "river surrounded by vegetation"
        return [0.1] * 512


class FakeVectorStore:
    def count(self):
        return 1

    def search(self, embedding, limit=5):
        assert len(embedding) == 512

        return [
            SimpleNamespace(
                id=1,
                score=0.85,
                payload={
                    "tile_id": "test_tile_001",
                    "source": "sample.tif",
                    "crs": "EPSG:4326",
                    "bounds": {
                        "left": 77.0,
                        "bottom": 28.9,
                        "right": 77.1,
                        "top": 29.0,
                    },
                    "width": 128,
                    "height": 128,
                    "resolution": [0.001, 0.001],
                    "path": "data/processed/test_tile_001.tif",
                },
            )
        ][:limit]


@pytest.fixture
def client():
    # Avoid loading RemoteCLIP or opening the real Qdrant database.
    app.state.embedder = FakeEmbedder()
    app.state.store = FakeVectorStore()

    # Do not use a "with" block: that would start the real app lifespan.
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client


def test_health_returns_indexed_tile_count(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["indexed_tiles"] == 1


def test_search_returns_matching_tile(client):
    response = client.post(
        "/search",
        json={
            "query": "river surrounded by vegetation",
            "limit": 4,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "river surrounded by vegetation"
    assert len(data["results"]) == 1

    tile = data["results"][0]
    assert tile["tile_id"] == "test_tile_001"
    assert tile["crs"] == "EPSG:4326"
    assert tile["center"]["latitude"] == pytest.approx(28.95)
    assert tile["center"]["longitude"] == pytest.approx(77.05)


def test_search_rejects_whitespace_query(client):
    response = client.post(
        "/search",
        json={"query": "   ", "limit": 4},
    )

    assert response.status_code == 422


def test_search_rejects_empty_query(client):
    response = client.post(
        "/search",
        json={"query": "", "limit": 4},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, -1, 21])
def test_search_rejects_invalid_limit(client, limit):
    response = client.post(
        "/search",
        json={
            "query": "river surrounded by vegetation",
            "limit": limit,
        },
    )

    assert response.status_code == 422