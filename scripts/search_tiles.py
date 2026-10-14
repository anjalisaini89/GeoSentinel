
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosentinel.embeddings.remoteclip import RemoteCLIPEmbedder
from geosentinel.embeddings.vector_store import QdrantVectorStore


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    checkpoint = (
        project_root
        / "models"
        / "remoteclip"
        / "RemoteCLIP-ViT-B-32.pt"
    )

    query = input(
        "Describe the satellite scene you want to find: "
    ).strip()

    if not query:
        print("Please enter a search description.")
        return

    embedder = RemoteCLIPEmbedder(
        checkpoint_path=checkpoint,
    )

    store = QdrantVectorStore(
        storage_path=project_root / "data" / "qdrant",
        collection_name="satellite_tiles",
        vector_size=512,
    )

    try:
        print(f"\nSearching for: {query}")
        print(f"Indexed tiles: {store.count()}")

        query_embedding = embedder.embed_text(query)
        results = store.search(query_embedding, limit=5)

        if not results:
            print("No matching tiles found.")
            return

        print("\nMost similar tiles:")
        for rank, result in enumerate(results, start=1):
            metadata = result.payload or {}
            tile_id = metadata.get(
                "tile_id",
                metadata.get("tile", str(result.id)),
            )

            print(f"\n{rank}. Tile: {tile_id}")
            print(f"   Similarity score: {result.score:.4f}")
            print(f"   Path: {metadata.get('path', 'Not available')}")
            print(f"   CRS: {metadata.get('crs', 'Not available')}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
