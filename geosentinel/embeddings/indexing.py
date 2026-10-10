
import json
from pathlib import Path
from typing import Any

from geosentinel.embeddings.pipeline import generate_embedding
from geosentinel.embeddings.remoteclip import RemoteCLIPEmbedder
from geosentinel.embeddings.vector_store import QdrantVectorStore


def index_tiles(
    manifest_path: str | Path,
    embedder: RemoteCLIPEmbedder,
    store: QdrantVectorStore,
) -> int:
    """
    Generate RemoteCLIP embeddings for manifest tiles
    and index them in Qdrant.

    Returns the number of tiles indexed.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest: list[dict[str, Any]] = json.load(file)

    if not isinstance(manifest, list):
        raise ValueError(
            "Manifest must contain a JSON list of tiles."
        )

    indexed = 0

    for tile in manifest:
        tile_path = Path(tile["path"])

        if not tile_path.is_absolute():
            tile_path = manifest_path.parent / tile_path

        embedding = generate_embedding(
            file_path=tile_path,
            embedder=embedder,
            expected_bands=3,
        )

        metadata = {
            key: value
            for key, value in tile.items()
            if key != "path"
        }
        metadata["path"] = str(tile_path.resolve())

        store.add_embedding(
            point_id=indexed + 1,
            embedding=embedding,
            metadata=metadata,
        )

        indexed += 1

    return indexed
