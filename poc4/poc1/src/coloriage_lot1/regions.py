"""Extraction des composantes connexes à partir des classes de palette."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class Region:
    region_id: int
    palette_index: int
    pixel_count: int
    pixel_percent: float
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    width: int
    height: int


def extract_regions(
    palette_labels: IntArray,
    palette_size: int,
    connectivity: int = 8,
) -> tuple[NDArray[np.uint32], list[Region]]:
    """Numérote les composantes de chaque classe de couleur."""
    if palette_labels.ndim != 2:
        raise ValueError("palette_labels doit être une matrice 2D")
    if connectivity not in (4, 8):
        raise ValueError("connectivity doit valoir 4 ou 8")

    structure = (
        ndimage.generate_binary_structure(2, 1)
        if connectivity == 4
        else ndimage.generate_binary_structure(2, 2)
    )
    height, width = palette_labels.shape
    total_pixels = int(height * width)
    global_labels = np.zeros((height, width), dtype=np.uint32)
    regions: list[Region] = []
    next_region_id = 1

    for palette_index in range(palette_size):
        local_labels, count = ndimage.label(
            palette_labels == palette_index,
            structure=structure,
        )
        if count == 0:
            continue

        objects = ndimage.find_objects(local_labels)
        for local_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            ys, xs = slices
            local_view = local_labels[ys, xs] == local_id
            pixel_count = int(np.count_nonzero(local_view))
            target = global_labels[ys, xs]
            target[local_view] = next_region_id
            min_y, max_y = int(ys.start), int(ys.stop - 1)
            min_x, max_x = int(xs.start), int(xs.stop - 1)
            regions.append(
                Region(
                    region_id=next_region_id,
                    palette_index=palette_index,
                    pixel_count=pixel_count,
                    pixel_percent=100.0 * pixel_count / total_pixels,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    width=max_x - min_x + 1,
                    height=max_y - min_y + 1,
                )
            )
            next_region_id += 1

    return global_labels, regions


def make_region_preview(region_labels: IntArray) -> NDArray[np.uint8]:
    """Crée une visualisation déterministe aux couleurs arbitraires."""
    ids = region_labels.astype(np.uint64)
    red = (ids * 67 + 29) % 223 + 24
    green = (ids * 131 + 47) % 223 + 24
    blue = (ids * 197 + 71) % 223 + 24
    preview = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    preview[ids == 0] = 255
    return preview

