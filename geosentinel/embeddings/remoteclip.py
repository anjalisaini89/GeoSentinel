from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip

from geosentinel.embeddings.base import ImageEmbedder


class RemoteCLIPEmbedder(ImageEmbedder):
    """
    RemoteCLIP ViT-B/32 image embedder.

    Uses the locally downloaded RemoteCLIP checkpoint and
    produces normalized 512-dimensional embeddings.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RemoteCLIP checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create the OpenAI-compatible ViT-B/32 architecture.
        self.model = open_clip.create_model(
            "ViT-B-32",
            pretrained="openai",
            device=self.device,
        )

        # Load the RemoteCLIP checkpoint.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "RemoteCLIP checkpoint must contain a state dictionary."
            )

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint,
            strict=False,
        )

        if missing_keys:
            raise RuntimeError(
                "RemoteCLIP checkpoint is missing model keys: "
                f"{missing_keys[:10]}"
            )

        if unexpected_keys:
            raise RuntimeError(
                "RemoteCLIP checkpoint contains unexpected keys: "
                f"{unexpected_keys[:10]}"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Use the preprocessing pipeline associated with ViT-B/32.
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )

    @property
    def embedding_dimension(self) -> int:
        """Return the RemoteCLIP embedding dimension."""
        return 512

    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a RemoteCLIP embedding from a CHW NumPy image.

        Parameters
        ----------
        image:
            Image array in CHW format with 3 channels.
            Values are expected to be in [0, 1].

        Returns
        -------
        np.ndarray
            L2-normalized 512-dimensional embedding.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array.")

        if image.ndim != 3:
            raise ValueError(
                "Expected image in CHW format: "
                "(channels, height, width)."
            )

        if image.shape[0] != 3:
            raise ValueError(
                "RemoteCLIP requires exactly 3 image channels."
            )

        if image.size == 0:
            raise ValueError("Image cannot be empty.")

        if not np.issubdtype(image.dtype, np.number):
            raise TypeError("Image must contain numeric values.")

        image = np.asarray(image, dtype=np.float32)
        image = np.clip(image, 0.0, 1.0)

        # CHW -> HWC.
        image = np.transpose(image, (1, 2, 0))

        # [0, 1] -> [0, 255].
        image_uint8 = (
            image * 255.0
        ).round().astype(np.uint8)

        pil_image = Image.fromarray(
            image_uint8,
            mode="RGB",
        )

        input_tensor = self.preprocess(
            pil_image
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embedding = self.model.encode_image(
                input_tensor
            )

        # L2-normalize for cosine-similarity retrieval.
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        return (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )