
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from geosentinel.embeddings.remoteclip import RemoteCLIPEmbedder
from geosentinel.embeddings.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "remoteclip"
    / "RemoteCLIP-ViT-B-32.pt"
)
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

_embedder: RemoteCLIPEmbedder | None = None
_store: QdrantVectorStore | None = None
_search_lock = Lock()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    tile_id: str
    score: float
    path: str | None = None
    crs: str | None = None
    bounds: dict[str, float] | None = None
    center: dict[str, float] | None = None
    width: int | None = None
    height: int | None = None
    resolution: list[float] | None = None
    source: str | None = None


class SearchResponse(BaseModel):
    query: str
    indexed_tiles: int
    results: list[SearchResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store

    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            f"RemoteCLIP checkpoint not found: {CHECKPOINT_PATH}"
        )

    _embedder = RemoteCLIPEmbedder(
        checkpoint_path=CHECKPOINT_PATH
    )
    _store = QdrantVectorStore(
        storage_path=QDRANT_PATH,
        collection_name="satellite_tiles",
        vector_size=512,
    )

    try:
        yield
    finally:
        if _store is not None:
            _store.close()
        _store = None
        _embedder = None


app = FastAPI(
    title="GeoSentinel API",
    description="Natural-language search over indexed satellite tiles.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _embedder is not None,
        "vector_store_connected": _store is not None,
        "indexed_tiles": _store.count() if _store is not None else 0,
    }


@app.post("/search", response_model=SearchResponse)
def search_tiles(request: SearchRequest) -> SearchResponse:
    if _embedder is None or _store is None:
        raise HTTPException(
            status_code=503,
            detail="Search service is not initialized.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query must not contain only whitespace.",
        )

    try:
        with _search_lock:
            total = _store.count()

            if total == 0:
                return SearchResponse(
                    query=query,
                    indexed_tiles=0,
                    results=[],
                )

            embedding = _embedder.embed_text(query)
            matches = _store.search(
                embedding,
                limit=request.limit,
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Search failed. Check the server logs.",
        ) from exc

    results: list[SearchResult] = []

    for match in matches:
        metadata = match.payload or {}
        bounds_data = metadata.get("bounds")
        bounds = None

        if isinstance(bounds_data, dict):
            try:
                bounds = {
                    key: float(bounds_data[key])
                    for key in ("left", "bottom", "right", "top")
                }
            except (KeyError, TypeError, ValueError):
                bounds = None

        center = None
        if bounds is not None and metadata.get("crs") == "EPSG:4326":
            center = {
                "longitude": (
                    bounds["left"] + bounds["right"]
                ) / 2,
                "latitude": (
                    bounds["bottom"] + bounds["top"]
                ) / 2,
            }

        resolution_data = metadata.get("resolution")
        resolution = None
        if isinstance(resolution_data, (list, tuple)):
            try:
                resolution = [float(value) for value in resolution_data]
            except (TypeError, ValueError):
                resolution = None

        results.append(
            SearchResult(
                tile_id=str(
                    metadata.get(
                        "tile_id",
                        metadata.get("tile", match.id),
                    )
                ),
                score=float(match.score),
                path=metadata.get("path"),
                crs=metadata.get("crs"),
                bounds=bounds,
                center=center,
                width=metadata.get("width"),
                height=metadata.get("height"),
                resolution=resolution,
                source=metadata.get("source"),
            )
        )

    return SearchResponse(
        query=query,
        indexed_tiles=total,
        results=results,
    )
