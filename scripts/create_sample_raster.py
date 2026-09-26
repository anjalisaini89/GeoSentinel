from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


OUTPUT_PATH = Path("data/samples/sample.tif")


def create_sample_raster() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    width = 256
    height = 256
    bands = 3

    # Synthetic RGB satellite-like data
    data = np.random.randint(
        0,
        255,
        size=(bands, height, width),
        dtype=np.uint8,
    )

    transform = from_origin(
        77.0,      # west
        29.0,      # north
        0.0001,    # pixel width
        0.0001,    # pixel height
    )

    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    print(f"Created sample raster: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_sample_raster()