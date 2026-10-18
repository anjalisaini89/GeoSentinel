
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import geosentinel.api.app as api_module


class FakeEmbedder:
    def embed_text(self, text):
        assert text == "river surrounded by vegetation"
        return [0.1] * 512


class FakeVectorStore:
    def count(self):
        return 1

    def search(self, embedding, limit=5):
        assert len(embedding) == 512

        results = [
            SimpleNamespace(
                id=1,
                score=0.85,
                payload={
                    "tile_id": "sample_r0_c0",
                    "path": "data/processed/sample_r0_c0.tif",
                    "crs": "EPSG:4326",
                    "bounds": {
                        "left": 77.0,
                        "bottom": 28.9872,
                        "right": 77.0128,
                        "top": 29.0,
                    },
                    "width": 128,
                    "height": 128,
                    "resolution": [0.0001, 0.0001],
                    "source": "synthetic",
                },
            )
        ]

        return results[:limit]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api_module, "_embedder", FakeEmbedder()
    )
    monkeypatch.setattr(
        api_module, "_store", FakeVectorStore()
    )

    # Do not use TestClient as a context manager here:
    # that would start the real application lifespan and
    # load the actual RemoteCLIP model.
    with_client = TestClient(
        api_module.app,
        raise_server_exceptions=True,
    )
    yield with_client
    with_client.close()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "vector_store_connected": True,
        "indexed_tiles": 1,
    }


def test_search_endpoint(client):
    response = client.post(
        "/search",
        json={
            "query": "river surrounded by vegetation",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "river surrounded by vegetation"
    assert data["indexed_tiles"] == 1
    assert len(data["results"]) == 1

    result = data["results"][0]
    assert result["tile_id"] == "sample_r0_c0"
    assert result["score"] == pytest.approx(0.85)
    assert result["crs"] == "EPSG:4326"
    assert result["center"]["longitude"] == pytest.approx(77.0064)
    assert result["center"]["latitude"] == pytest.approx(28.9936)
    assert result["width"] == 128
    assert result["height"] == 128


def test_search_rejects_empty_query(client):
    response = client.post(
        "/search",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_search_rejects_whitespace_only_query(client):
    response = client.post(
        "/search",
        json={"query": "   "},
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
