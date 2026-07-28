"""Perceptual palette cleanup and global palette construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.color import deltaE_ciede2000


@dataclass(frozen=True)
class GlobalPaletteResult:
    centers_lab: NDArray[np.float64]
    labels: NDArray[np.int32]
    metadata: dict[str, Any]


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
    merged, remapped = _stable_order(merged, remapped)
    return merged, remapped, {
        "before": int(len(centers)),
        "after": int(len(merged)),
        "merged": int(len(centers) - len(merged)),
        "threshold": float(threshold),
    }


def _stable_order(
    centers: NDArray[np.float64],
    labels: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    if len(centers) == 0:
        return centers, labels.astype(np.int32)
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return centers[order], inverse[labels].astype(np.int32)


def _assign_nearest(
    pixels_lab: NDArray[np.float64],
    centers_lab: NDArray[np.float64],
    *,
    chunk_size: int = 100_000,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    labels = np.empty(len(pixels_lab), dtype=np.int32)
    squared_error = np.empty(len(pixels_lab), dtype=np.float64)
    for start in range(0, len(pixels_lab), chunk_size):
        stop = min(len(pixels_lab), start + chunk_size)
        chunk = pixels_lab[start:stop]
        distances = np.sum((chunk[:, None, :] - centers_lab[None, :, :]) ** 2, axis=2)
        local = np.argmin(distances, axis=1)
        labels[start:stop] = local.astype(np.int32)
        squared_error[start:stop] = distances[np.arange(stop - start), local]
    return labels, squared_error


def _recompute_centers(
    pixels_lab: NDArray[np.float64],
    labels: NDArray[np.int32],
    count: int,
    importance_weights: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    weights = (
        np.ones(len(pixels_lab), dtype=np.float64)
        if importance_weights is None
        else np.asarray(importance_weights, dtype=np.float64)
    )
    centers = np.zeros((count, 3), dtype=np.float64)
    totals = np.bincount(labels, weights=weights, minlength=count)
    for channel in range(3):
        sums = np.bincount(
            labels,
            weights=pixels_lab[:, channel] * weights,
            minlength=count,
        )
        centers[:, channel] = sums / np.maximum(totals, 1e-12)
    return centers


def _refill_exact_palette(
    pixels_lab: NDArray[np.float64],
    centers_lab: NDArray[np.float64],
    requested_colors: int,
    importance_weights: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.int32], int]:
    centers = np.asarray(centers_lab, dtype=np.float64).copy()
    weights = (
        np.ones(len(pixels_lab), dtype=np.float64)
        if importance_weights is None
        else np.asarray(importance_weights, dtype=np.float64)
    )
    replenished = 0
    attempts = 0
    max_attempts = max(8, requested_colors * 4)

    while attempts < max_attempts:
        labels, error = _assign_nearest(pixels_lab, centers)
        used = np.unique(labels)
        if len(used) != len(centers):
            remap = np.full(len(centers), -1, dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            centers = centers[used]
            labels = remap[labels]
            labels, error = _assign_nearest(pixels_lab, centers)
        if len(centers) >= requested_colors:
            return centers[:requested_colors], labels, replenished

        score = error * np.maximum(weights, 0.01)
        for existing in centers:
            duplicate = np.all(
                np.isclose(pixels_lab, existing[None, :], atol=1e-7),
                axis=1,
            )
            score[duplicate] = -1.0
        index = int(np.argmax(score))
        if score[index] <= 1e-12:
            break
        centers = np.vstack([centers, pixels_lab[index]])
        replenished += 1
        attempts += 1

    labels, _ = _assign_nearest(pixels_lab, centers)
    used = np.unique(labels)
    if len(used) != len(centers):
        remap = np.full(len(centers), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        centers = centers[used]
        labels = remap[labels]
    return centers, labels, replenished


def build_global_palette(
    pixels_lab: NDArray[np.float64],
    initial_centers_lab: NDArray[np.float64],
    initial_labels: NDArray[np.int32],
    *,
    requested_colors: int,
    mode: str = "adaptive",
    merge_threshold: float = 4.0,
    importance_weights: NDArray[np.float64] | None = None,
) -> GlobalPaletteResult:
    """Build one palette shared by subject and background.

    ``adaptive`` treats requested_colors as a maximum. ``exact`` replenishes
    perceptually useful colors after duplicate removal. ``legacy`` only keeps
    the supplied candidate palette and is used for v1 comparisons.
    """
    if mode not in {"legacy", "adaptive", "exact"}:
        raise ValueError("palette_mode doit valoir legacy, adaptive ou exact")
    pixels = np.asarray(pixels_lab, dtype=np.float64)
    centers = np.asarray(initial_centers_lab, dtype=np.float64)
    labels = np.asarray(initial_labels, dtype=np.int32)
    if len(pixels) != len(labels):
        raise ValueError("initial_labels doit couvrir tous les pixels")
    if importance_weights is not None and len(importance_weights) != len(pixels):
        raise ValueError("importance_weights doit couvrir tous les pixels")

    if mode == "legacy":
        ordered, remapped = _stable_order(centers, labels)
        return GlobalPaletteResult(
            centers_lab=ordered,
            labels=remapped,
            metadata={
                "mode": mode,
                "before": int(len(centers)),
                "after": int(len(ordered)),
                "merged": 0,
                "replenished": 0,
                "requested": int(requested_colors),
            },
        )

    merged, _, cleanup = merge_near_palette_colors(
        centers,
        labels,
        threshold=merge_threshold,
        minimum_colors=min(2, requested_colors),
    )
    reassigned, _ = _assign_nearest(pixels, merged)
    merged = _recompute_centers(pixels, reassigned, len(merged), importance_weights)
    reassigned, _ = _assign_nearest(pixels, merged)
    replenished = 0

    if mode == "exact" and len(merged) < requested_colors:
        merged, reassigned, replenished = _refill_exact_palette(
            pixels,
            merged,
            requested_colors,
            importance_weights,
        )

    used = np.unique(reassigned)
    if len(used) != len(merged):
        remap = np.full(len(merged), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        merged = merged[used]
        reassigned = remap[reassigned]
    merged, reassigned = _stable_order(merged, reassigned)
    return GlobalPaletteResult(
        centers_lab=merged,
        labels=reassigned,
        metadata={
            "mode": mode,
            "before": int(len(centers)),
            "after": int(len(merged)),
            "merged": int(cleanup["merged"]),
            "replenished": int(replenished),
            "requested": int(requested_colors),
            "threshold": float(merge_threshold),
            "exact_achieved": bool(mode != "exact" or len(merged) == requested_colors),
        },
    )
