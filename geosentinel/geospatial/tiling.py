from pathlib import Path

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


def tile_raster(
    input_path: str | Path,
    output_dir: str | Path,
    tile_size: int = 256,
) -> list[Path]:
    """
    Split a raster into square GeoTIFF tiles.

    Each tile preserves the source CRS and geospatial transform.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {input_path}"
        )

    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_tiles: list[Path] = []

    with rasterio.open(input_path) as src:

        for row in range(0, src.height, tile_size):
            for col in range(0, src.width, tile_size):

                width = min(tile_size, src.width - col)
                height = min(tile_size, src.height - row)

                window = Window(
                    col_off=col,
                    row_off=row,
                    width=width,
                    height=height,
                )

                data = src.read(window=window)

                tile_transform = window_transform(
                    window,
                    src.transform,
                )

                tile_path = (
                    output_dir
                    / f"{input_path.stem}_r{row}_c{col}.tif"
                )

                profile = src.profile.copy()

                profile.update(
                    {
                        "height": height,
                        "width": width,
                        "transform": tile_transform,
                    }
                )

                with rasterio.open(
                    tile_path,
                    "w",
                    **profile,
                ) as dst:
                    dst.write(data)

                generated_tiles.append(tile_path)

    return generated_tiles