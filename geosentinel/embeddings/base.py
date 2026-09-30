from abc import ABC, abstractmethod

import numpy as np


class ImageEmbedder(ABC):
    """
    Base interface for image embedding models.

    Any future embedding model such as RemoteCLIP
    should implement this interface.
    """

    @abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image into a vector embedding.

        Parameters
        ----------
        image:
            Image array in CHW format.

        Returns
        -------
        np.ndarray
            One-dimensional embedding vector.
        """
        raise NotImplementedError


class DummyEmbedder(ImageEmbedder):
    """
    Temporary deterministic embedder for pipeline testing.

    This is NOT a real AI embedding model.
    It allows us to test the embedding pipeline before
    installing a large remote-sensing model.
    """

    def __init__(self, embedding_dimension: int = 128):
        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero."
            )

        self.embedding_dimension = embedding_dimension

    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a deterministic test embedding from image statistics.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array.")

        if image.ndim != 3:
            raise ValueError(
                "Expected image in CHW format: "
                "(channels, height, width)."
            )

        if image.size == 0:
            raise ValueError("Image cannot be empty.")

        if not np.issubdtype(image.dtype, np.number):
            raise TypeError("Image must contain numeric values.")

        # Basic deterministic statistics.
        statistics = np.array(
            [
                float(image.mean()),
                float(image.std()),
                float(image.min()),
                float(image.max()),
            ],
            dtype=np.float32,
        )

        # Repeat statistics to create a fixed-size vector.
        embedding = np.resize(
            statistics,
            self.embedding_dimension,
        ).astype(np.float32)

        # L2-normalize the vector.
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding /= norm

        return embedding