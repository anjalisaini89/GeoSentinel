from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip

from geosentinel.embeddings.base import ImageEmbedder


class RemoteCLIPEmbedder(ImageEmbedder):
    """
    RemoteCLIP image embedder for satellite imagery.

    The model runs locally using a downloaded RemoteCLIP
    ViT-B/32 checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"RemoteCLIP checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=None,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def embedding_dimension(self) -> int:
        """Return the RemoteCLIP embedding dimension."""
        return 512

    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a RemoteCLIP embedding from a CHW NumPy image.
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

        # Expected input range is [0, 1].
        image = np.clip(image, 0.0, 1.0)

        # CHW -> HWC.
        image = np.transpose(image, (1, 2, 0))

        # Float [0, 1] -> uint8 [0, 255].
        image_uint8 = (image * 255.0).round().astype(np.uint8)

        pil_image = Image.fromarray(image_uint8, mode="RGB")

        input_tensor = self.preprocess(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.inference_mode():
            embedding = self.model.encode_image(input_tensor)

        # L2 normalization is standard for CLIP-style embeddings.
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)