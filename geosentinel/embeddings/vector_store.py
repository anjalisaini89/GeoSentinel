from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


class QdrantVectorStore:
    """
    Local Qdrant vector store for GeoSentinel embeddings.

    Stores normalized RemoteCLIP embeddings and associated
    satellite tile metadata.
    """

    def __init__(
        self,
        storage_path: str | Path = "data/qdrant",
        collection_name: str = "satellite_tiles",
        vector_size: int = 512,
    ):
        if vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than zero."
            )

        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=str(self.storage_path)
        )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""

        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_embedding(
        self,
        point_id: int,
        embedding: np.ndarray,
        metadata: dict,
    ) -> None:
        """
        Store one embedding with its metadata.
        """

        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                "embedding must be a NumPy array."
            )

        if embedding.ndim != 1:
            raise ValueError(
                "embedding must be a 1D vector."
            )

        if embedding.shape[0] != self.vector_size:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.vector_size}, "
                f"got {embedding.shape[0]}."
            )

        if not np.issubdtype(
            embedding.dtype,
            np.number,
        ):
            raise TypeError(
                "embedding must contain numeric values."
            )

        vector = embedding.astype(
            np.float32
        ).tolist()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata,
                )
            ],
        )

    def count(self) -> int:
        """Return the number of stored vectors."""

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def search(
        self,
        embedding: np.ndarray,
        limit: int = 5,
    ) -> list:
        """
        Find the most similar stored embeddings.

        Parameters
        ----------
        embedding:
            Query embedding as a 1D NumPy array.

        limit:
            Maximum number of results.

        Returns
        -------
        list
            Qdrant search results ordered by similarity.
        """

        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                "embedding must be a NumPy array."
            )

        if embedding.ndim != 1:
            raise ValueError(
                "embedding must be a 1D vector."
            )

        if embedding.shape[0] != self.vector_size:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.vector_size}, "
                f"got {embedding.shape[0]}."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        vector = embedding.astype(
            np.float32
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        return results.points

    def close(self) -> None:
        """Close the local Qdrant client."""

        self.client.close()