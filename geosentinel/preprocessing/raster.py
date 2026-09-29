from pathlib import Path

import numpy as np
import rasterio


def load_raster_array(
    file_path: str | Path,
    expected_bands: int | None = None,
) -> np.ndarray:
    """
    Load a raster into a NumPy array.

    Returns
    -------
    np.ndarray
        Array in CHW format:
        (bands, height, width)
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {file_path}"
        )

    with rasterio.open(file_path) as src:
        if expected_bands is not None and src.count != expected_bands:
            raise ValueError(
                f"Expected {expected_bands} bands, "
                f"but raster contains {src.count} bands."
            )

        data = src.read()

    return data


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    """
    Normalize uint8 pixel values from [0, 255] to [0, 1].
    """

    if array.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 array, got {array.dtype}."
        )

    return array.astype(np.float32) / 255.0


def preprocess_raster(
    file_path: str | Path,
    expected_bands: int = 3,
) -> np.ndarray:
    """
    Load and normalize a raster for downstream ML processing.

    Returns
    -------
    np.ndarray
        Float32 array in CHW format with values in [0, 1].
    """

    array = load_raster_array(
        file_path,
        expected_bands=expected_bands,
    )

    return normalize_uint8(array)