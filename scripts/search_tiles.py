
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosentinel.embeddings.remoteclip import RemoteCLIPEmbedder
from geosentinel.embeddings.vector_store import QdrantVectorStore


def format_bounds(bounds: dict) -> str:
    """Format a geographic bounding box safely."""
    if not isinstance(bounds, dict):
        return "Not available"

    required = ("left", "bottom", "right", "top")

    if not all(key in bounds for key in required):
        return "Not available"

    try:
        west, south, east, north = (
            float(bounds[key]) for key in required
        )
    except (TypeError, ValueError):
        return "Invalid bounds"

    return (
        f"West={west:.6f}, South={south:.6f}, "
        f"East={east:.6f}, North={north:.6f}"
    )


def format_center(bounds: dict, crs: str) -> str:
    """Calculate the center for geographic coordinates."""
    if crs != "EPSG:4326" or not isinstance(bounds, dict):
        return "Unavailable (requires EPSG:4326)"

    try:
        longitude = (
            float(bounds["left"]) + float(bounds["right"])
        ) / 2
        latitude = (
            float(bounds["bottom"]) + float(bounds["top"])
        ) / 2
    except (KeyError, TypeError, ValueError):
        return "Not available"

    return f"Latitude={latitude:.6f}, Longitude={longitude:.6f}"


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
        total = store.count()

        print(f"\nSearching for: {query}")
        print(f"Indexed tiles: {total}")

        if total == 0:
            print("No tiles are indexed yet.")
            return

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

            bounds = metadata.get("bounds", {})
            crs = metadata.get("crs", "Not available")
            resolution = metadata.get("resolution")
            width = metadata.get("width", "Unknown")
            height = metadata.get("height", "Unknown")

            print(f"\n{'=' * 55}")
            print(f"{rank}. Tile: {tile_id}")
            print(f"   Similarity score: {result.score:.4f}")
            print(f"   CRS: {crs}")
            print(f"   Bounding box: {format_bounds(bounds)}")
            print(f"   Center: {format_center(bounds, crs)}")
            print(f"   Dimensions: {width} x {height} pixels")
            print(f"   Resolution: {resolution or 'Not available'}")
            print(f"   Source: {metadata.get('source', 'Not available')}")
            print(f"   Path: {metadata.get('path', 'Not available')}")

        print(f"\n{'=' * 55}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
