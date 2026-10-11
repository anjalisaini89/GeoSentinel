
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosentinel.embeddings.indexing import index_tiles
from geosentinel.embeddings.remoteclip import RemoteCLIPEmbedder
from geosentinel.embeddings.vector_store import QdrantVectorStore


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    checkpoint = (
        project_root / "models" / "remoteclip" / "RemoteCLIP-ViT-B-32.pt"
    )
    manifest = project_root / "data" / "processed" / "manifest.json"

    embedder = RemoteCLIPEmbedder(checkpoint_path=checkpoint)

    store = QdrantVectorStore(
        storage_path=project_root / "data" / "qdrant",
        collection_name="satellite_tiles",
        vector_size=512,
    )

    try:
        print("Starting GeoSentinel tile indexing...")

        indexed = index_tiles(
            manifest_path=manifest,
            embedder=embedder,
            store=store,
        )

        print(f"Tiles processed: {indexed}")
        print(f"Total vectors in Qdrant: {store.count()}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
