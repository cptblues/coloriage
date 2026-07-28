"""Perceptual palette cleanup helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.color import deltaE_ciede2000


def merge_near_palette_colors(
    centers_lab: NDArray[np.float64],
    labels: NDArray[np.int32],
    *,
    threshold: float = 4.0,
    minimum_colors: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.int32], dict[str, int | float]]:
    """Greedily merge low-usage colors into perceptually close dominant colors."""
    centers = np.asarray(centers_lab, dtype=np.float64)
    flat_labels = np.asarray(labels, dtype=np.int32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_lab doit être de forme (n, 3)")
    if len(centers) <= minimum_colors or threshold <= 0:
        return centers, flat_labels, {
            "before": int(len(centers)),
            "after": int(len(centers)),
            "merged": 0,
            "threshold": float(threshold),
        }

    counts = np.bincount(flat_labels.ravel(), minlength=len(centers)).astype(np.int64)
    dominant_order = np.argsort(-counts, kind="stable")
    groups: list[list[int]] = []
    representatives: list[int] = []

    for palette_index in dominant_order:
        palette_index = int(palette_index)
        best_group = -1
        best_delta = float("inf")
        for group_index, representative in enumerate(representatives):
            delta = float(
                deltaE_ciede2000(
                    centers[palette_index][None, :],
                    centers[representative][None, :],
                )[0]
            )
            if delta < best_delta:
                best_delta = delta
                best_group = group_index
        can_merge = (
            best_group >= 0
            and best_delta < threshold
            and len(groups) >= minimum_colors
        )
        if can_merge:
            groups[best_group].append(palette_index)
        else:
            representatives.append(palette_index)
            groups.append([palette_index])

    # If the greedy order temporarily protected too many colors, perform a
    # second safe pass without dropping below minimum_colors.
    changed = True
    while changed and len(groups) > minimum_colors:
        changed = False
        best_pair: tuple[float, int, int] | None = None
        weighted_centers = []
        for group in groups:
            weights = np.maximum(1, counts[group]).astype(np.float64)
            weighted_centers.append(np.average(centers[group], axis=0, weights=weights))
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                delta = float(
                    deltaE_ciede2000(
                        weighted_centers[left][None, :],
                        weighted_centers[right][None, :],
                    )[0]
                )
                if delta < threshold and (best_pair is None or delta < best_pair[0]):
                    best_pair = (delta, left, right)
        if best_pair is not None:
            _, left, right = best_pair
            groups[left].extend(groups[right])
            groups.pop(right)
            changed = True

    merged_centers: list[NDArray[np.float64]] = []
    old_to_group = np.zeros(len(centers), dtype=np.int32)
    for group_index, group in enumerate(groups):
        weights = np.maximum(1, counts[group]).astype(np.float64)
        merged_centers.append(np.average(centers[group], axis=0, weights=weights))
        old_to_group[group] = group_index

    merged = np.asarray(merged_centers, dtype=np.float64)
    remapped = old_to_group[flat_labels]
    order = np.lexsort((merged[:, 2], merged[:, 1], merged[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    merged = merged[order]
    remapped = inverse[remapped].astype(np.int32)
    return merged, remapped, {
        "before": int(len(centers)),
        "after": int(len(merged)),
        "merged": int(len(centers) - len(merged)),
        "threshold": float(threshold),
    }
