"""Segmentation SLIC compacte et lissage de la carte de palette."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .color import rgb_to_lab


def _slic_labels(
    normalized_rgb: NDArray[np.uint8],
    target_segments: int,
    compactness: float,
    iterations: int = 6,
) -> NDArray[np.int32]:
    """Implémentation locale du cœur de SLIC, sans dépendance supplémentaire."""
    height, width = normalized_rgb.shape[:2]
    pixel_count = height * width
    step = max(2.0, np.sqrt(pixel_count / target_segments))
    y_positions = np.arange(step / 2.0, height, step)
    x_positions = np.arange(step / 2.0, width, step)
    if not len(y_positions):
        y_positions = np.array([height / 2.0])
    if not len(x_positions):
        x_positions = np.array([width / 2.0])
    grid_y, grid_x = np.meshgrid(y_positions, x_positions, indexing="ij")
    center_y = np.clip(np.rint(grid_y.ravel()), 0, height - 1).astype(np.int32)
    center_x = np.clip(np.rint(grid_x.ravel()), 0, width - 1).astype(np.int32)
    lab = rgb_to_lab(normalized_rgb)
    centers = np.column_stack(
        [
            lab[center_y, center_x],
            center_y.astype(np.float64),
            center_x.astype(np.float64),
        ]
    )
    labels = np.full((height, width), -1, dtype=np.int32)
    spatial_weight = (compactness / step) ** 2
    yy, xx = np.mgrid[:height, :width]

    for _ in range(iterations):
        distances = np.full((height, width), np.inf, dtype=np.float64)
        for center_id, center in enumerate(centers):
            cy, cx = center[3], center[4]
            y0 = max(0, int(cy - 2.0 * step))
            y1 = min(height, int(cy + 2.0 * step) + 1)
            x0 = max(0, int(cx - 2.0 * step))
            x1 = min(width, int(cx + 2.0 * step) + 1)
            color_delta = lab[y0:y1, x0:x1] - center[:3]
            color_distance = np.sum(color_delta**2, axis=-1)
            spatial_distance = (
                (yy[y0:y1, x0:x1] - cy) ** 2
                + (xx[y0:y1, x0:x1] - cx) ** 2
            )
            candidate = color_distance + spatial_weight * spatial_distance
            current = distances[y0:y1, x0:x1]
            update = candidate < current
            current[update] = candidate[update]
            labels_view = labels[y0:y1, x0:x1]
            labels_view[update] = center_id

        flat_labels = labels.ravel()
        valid = flat_labels >= 0
        counts = np.bincount(
            flat_labels[valid],
            minlength=len(centers),
        ).astype(np.float64)
        nonempty = counts > 0
        for channel in range(3):
            sums = np.bincount(
                flat_labels[valid],
                weights=lab[..., channel].ravel()[valid],
                minlength=len(centers),
            )
            centers[nonempty, channel] = sums[nonempty] / counts[nonempty]
        for axis, coordinates in ((3, yy), (4, xx)):
            sums = np.bincount(
                flat_labels[valid],
                weights=coordinates.ravel()[valid],
                minlength=len(centers),
            )
            centers[nonempty, axis] = sums[nonempty] / counts[nonempty]
    return labels


def segment_palette_labels(
    normalized_rgb: NDArray[np.uint8],
    palette_labels: NDArray[np.int32],
    palette_size: int,
    method: str,
    superpixels: int,
    compactness: float,
    smoothing_radius: int,
) -> NDArray[np.int32]:
    """Produit une carte de couleurs spatialement cohérente."""
    if method not in ("components", "slic"):
        raise ValueError("segmentation doit valoir components ou slic")

    if method == "components":
        segmented = palette_labels.copy()
    else:
        slic_labels = _slic_labels(
            normalized_rgb,
            target_segments=superpixels,
            compactness=compactness,
        )
        superpixel_count = int(slic_labels.max()) + 1
        combined = slic_labels.ravel() * palette_size + palette_labels.ravel()
        votes = np.bincount(
            combined,
            minlength=superpixel_count * palette_size,
        ).reshape(superpixel_count, palette_size)
        assignments = np.argmax(votes, axis=1).astype(np.int32)
        segmented = assignments[slic_labels]

    if smoothing_radius > 0:
        window = 2 * smoothing_radius + 1
        votes = [
            ndimage.uniform_filter(
                (segmented == palette_index).astype(np.float32),
                size=window,
                mode="nearest",
            )
            for palette_index in range(palette_size)
        ]
        segmented = np.argmax(np.stack(votes, axis=0), axis=0).astype(np.int32)
    return segmented
