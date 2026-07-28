#!/usr/bin/env python3
"""Upgrade the Nuance/coloriage processing pipeline to clean-render v1.

Target repository: https://github.com/cptblues/coloriage
Expected baseline: commit 88864adc134f27cecce36717ac70437a2602463a
("feat: upgrade processing img")

Run from the repository root:

    python apply_clean_render_v1.py

Useful options:

    python apply_clean_render_v1.py --check
    python apply_clean_render_v1.py --no-tests
    python apply_clean_render_v1.py --repo /path/to/coloriage
    python apply_clean_render_v1.py --force

The migration:
- adds edge-preserving preprocessing;
- replaces the default home-grown SLIC path with scikit-image SLICO;
- merges perceptually redundant palette colors;
- performs extra merge passes for thin/non-labelable regions;
- switches SVG fills to subpixel contours;
- draws shared region boundaries once;
- overlays multiscale internal line-art details;
- keeps all numbers inside regions, with tiny fonts only as a last resort;
- adds focused unit tests and documentation.

The script creates backups, does not create a commit, and does not push.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_COMMIT = "88864adc134f27cecce36717ac70437a2602463a"
MARKER = "clean-render-v1"


PREPROCESSING_MODULE = r'''"""Edge-preserving preparation used before palette fitting and segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage import color, filters, restoration


def _resize_mask(mask: NDArray[np.bool_] | None, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    if mask.shape != shape:
        raise ValueError("Le masque doit avoir la même taille que l'image normalisée")
    return np.asarray(mask, dtype=bool)


def prepare_processing_image(
    rgb: NDArray[np.uint8],
    *,
    subject_mask: NDArray[np.bool_] | None = None,
    detail_mask: NDArray[np.bool_] | None = None,
    sigma_color: float = 0.055,
    sigma_spatial: float = 3.0,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """Reduce photographic texture while preserving meaningful boundaries.

    The subject receives a moderate bilateral denoise. The background receives
    a stronger second pass. User-painted detail zones blend more of the source
    image back in so eyes, faces, fur, and small objects retain structure.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb doit être une image RGB")
    if not 0.001 <= sigma_color <= 0.25:
        raise ValueError("sigma_color doit être compris entre 0,001 et 0,25")
    if not 0.25 <= sigma_spatial <= 12.0:
        raise ValueError("sigma_spatial doit être compris entre 0,25 et 12")

    shape = rgb.shape[:2]
    subject_mask = _resize_mask(subject_mask, shape)
    detail_mask = _resize_mask(detail_mask, shape)
    source = rgb.astype(np.float32) / 255.0

    base = restoration.denoise_bilateral(
        source,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        bins=256,
        channel_axis=-1,
    )
    background = restoration.denoise_bilateral(
        base,
        sigma_color=min(0.25, sigma_color * 1.45),
        sigma_spatial=min(12.0, sigma_spatial * 1.65),
        bins=256,
        channel_axis=-1,
    )

    # Stabilize chroma more than lightness to avoid palette speckle while
    # retaining the geometry carried by luminance boundaries.
    lab = color.rgb2lab(np.clip(base, 0.0, 1.0))
    lab[..., 1] = filters.gaussian(lab[..., 1], sigma=0.65, preserve_range=True)
    lab[..., 2] = filters.gaussian(lab[..., 2], sigma=0.65, preserve_range=True)
    base = np.clip(color.lab2rgb(lab), 0.0, 1.0)

    output = background
    if subject_mask is not None:
        output = np.where(subject_mask[..., None], base, background)
    else:
        output = 0.70 * base + 0.30 * background

    if detail_mask is not None:
        detailed = 0.68 * source + 0.32 * base
        output = np.where(detail_mask[..., None], detailed, output)

    result = np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)
    return result, {
        "mode": "edge_preserving",
        "sigma_color": float(sigma_color),
        "sigma_spatial": float(sigma_spatial),
        "subject_aware": subject_mask is not None,
        "detail_aware": detail_mask is not None,
    }
'''


PALETTE_MODULE = r'''"""Perceptual palette cleanup helpers."""

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
'''


LINEART_MODULE = r'''"""Multiscale internal line-art extraction and vector tracing."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage import color, feature, morphology, restoration

Point = tuple[float, float]
Pixel = tuple[int, int]


def _region_boundary_mask(region_labels: NDArray[np.integer]) -> NDArray[np.bool_]:
    boundary = np.zeros(region_labels.shape, dtype=bool)
    vertical = region_labels[:, :-1] != region_labels[:, 1:]
    boundary[:, :-1] |= vertical
    boundary[:, 1:] |= vertical
    horizontal = region_labels[:-1, :] != region_labels[1:, :]
    boundary[:-1, :] |= horizontal
    boundary[1:, :] |= horizontal
    return boundary


def build_line_art_mask(
    rgb: NDArray[np.uint8],
    *,
    subject_mask: NDArray[np.bool_] | None = None,
    detail_mask: NDArray[np.bool_] | None = None,
    region_labels: NDArray[np.integer] | None = None,
    detail_strength: float = 0.65,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Extract useful internal strokes without creating additional color zones."""
    if not 0.0 <= detail_strength <= 1.0:
        raise ValueError("detail_strength doit être compris entre 0 et 1")
    source = rgb.astype(np.float32) / 255.0
    gray = color.rgb2gray(source)
    gray = restoration.denoise_bilateral(
        gray,
        sigma_color=0.045,
        sigma_spatial=2.0,
        bins=256,
        channel_axis=None,
    )

    fine = feature.canny(
        gray,
        sigma=1.05,
        low_threshold=0.045,
        high_threshold=0.16,
        use_quantiles=False,
    )
    coarse = feature.canny(
        gray,
        sigma=2.15,
        low_threshold=0.035,
        high_threshold=0.13,
        use_quantiles=False,
    )

    if subject_mask is not None:
        subject = np.asarray(subject_mask, dtype=bool)
        subject = morphology.binary_dilation(subject, morphology.disk(1))
        combined = coarse | (fine & subject)
    else:
        combined = coarse | (fine if detail_strength >= 0.55 else False)

    if detail_mask is not None:
        details = morphology.binary_dilation(
            np.asarray(detail_mask, dtype=bool),
            morphology.disk(2),
        )
        combined |= fine & details

    if region_labels is not None:
        boundaries = morphology.binary_dilation(
            _region_boundary_mask(region_labels),
            morphology.disk(1),
        )
        combined &= ~boundaries

    min_size = max(7, int(round(combined.size / 180_000)))
    try:
        combined = morphology.remove_small_objects(
            combined,
            max_size=max(0, min_size - 1),
        )
    except TypeError:  # scikit-image < 0.26
        combined = morphology.remove_small_objects(combined, min_size=min_size)
    skeleton = morphology.skeletonize(combined)
    return np.asarray(skeleton, dtype=bool), {
        "enabled": True,
        "detail_strength": float(detail_strength),
        "stroke_pixels": int(np.count_nonzero(skeleton)),
        "component_min_size": int(min_size),
        "subject_aware": subject_mask is not None,
        "detail_aware": detail_mask is not None,
    }


def trace_skeleton_polylines(mask: NDArray[np.bool_]) -> list[list[Point]]:
    """Trace an 8-connected skeleton into open polylines and closed cycles."""
    pixels = {tuple(int(v) for v in point) for point in np.argwhere(mask)}
    if not pixels:
        return []
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    graph: dict[Pixel, list[Pixel]] = {}
    for y, x in pixels:
        graph[(y, x)] = [
            (y + dy, x + dx)
            for dy, dx in offsets
            if (y + dy, x + dx) in pixels
        ]

    def edge(a: Pixel, b: Pixel) -> frozenset[Pixel]:
        return frozenset((a, b))

    unused = {
        edge(node, neighbor)
        for node, neighbors in graph.items()
        for neighbor in neighbors
        if node != neighbor
    }
    paths: list[list[Pixel]] = []

    def walk(start: Pixel, next_node: Pixel) -> list[Pixel]:
        path = [start, next_node]
        unused.discard(edge(start, next_node))
        previous = start
        current = next_node
        while True:
            candidates = [
                candidate
                for candidate in graph[current]
                if candidate != previous and edge(current, candidate) in unused
            ]
            if not candidates or (len(graph[current]) != 2 and current != start):
                break
            candidate = min(candidates)
            unused.discard(edge(current, candidate))
            previous, current = current, candidate
            path.append(current)
            if current == start:
                break
        return path

    endpoints = sorted(node for node, neighbors in graph.items() if len(neighbors) != 2)
    for node in endpoints:
        for neighbor in sorted(graph[node]):
            if edge(node, neighbor) in unused:
                paths.append(walk(node, neighbor))

    while unused:
        first_edge = next(iter(unused))
        start, neighbor = tuple(first_edge)
        paths.append(walk(start, neighbor))

    output: list[list[Point]] = []
    for path in paths:
        if len(path) < 3:
            continue
        output.append([(x + 0.5, y + 0.5) for y, x in path])
    return output
'''


QUALITY_MODULE = r'''"""Post-segmentation cleanup focused on printable and labelable regions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .regions import MergeEvent, describe_regions, merge_small_regions


@dataclass(frozen=True)
class CleanMergeResult:
    region_labels: NDArray[np.uint32]
    region_palette: NDArray[np.int32]
    events: list[MergeEvent]
    forced_merges: int
    passes_executed: int


def _region_groups(
    region_labels: NDArray[np.uint32],
    subject_mask: NDArray[np.bool_] | None,
) -> NDArray[np.int8] | None:
    if subject_mask is None:
        return None
    count = int(region_labels.max()) + 1
    total = np.bincount(region_labels.ravel(), minlength=count)
    subject = np.bincount(
        region_labels[np.asarray(subject_mask, dtype=bool)].ravel(),
        minlength=count,
    )
    groups = np.zeros(count, dtype=np.int8)
    groups[1:] = (subject[1:] * 2 >= total[1:]).astype(np.int8)
    return groups


def improve_region_labelability(
    *,
    region_labels: NDArray[np.uint32],
    region_palette: NDArray[np.int32],
    palette_lab: NDArray[np.float64],
    mm_per_pixel: float,
    min_region_pixels: int,
    min_region_area_mm2: float,
    preferred_font_mm: float,
    min_font_mm: float,
    padding_mm: float,
    strategy: str,
    color_tolerance: float,
    subject_mask: NDArray[np.bool_] | None,
    passes: int = 2,
) -> CleanMergeResult:
    """Merge thin strips before falling back to microscopic label fonts."""
    current_labels = np.asarray(region_labels, dtype=np.uint32)
    current_palette = np.asarray(region_palette, dtype=np.int32)
    all_events: list[MergeEvent] = []
    forced_merges = 0
    passes_executed = 0

    for _ in range(max(0, passes)):
        palette_digits = max(1, len(str(max(1, len(palette_lab)))))
        global_required_mm = min_font_mm * max(1.0, 0.62 * palette_digits) + 2.0 * padding_mm
        regions = describe_regions(
            current_labels,
            current_palette,
            mm_per_pixel,
            min_region_pixels,
            global_required_mm,
        )
        count = int(current_labels.max()) + 1
        thresholds = np.full(count, min_region_pixels, dtype=np.int64)
        needs_merge = 0
        for region in regions:
            number = int(current_palette[region.region_id]) + 1
            digit_factor = max(1.0, 0.62 * len(str(number)))
            required_width = min_font_mm * digit_factor + 2.0 * padding_mm
            awkward = (
                region.compactness < 0.018
                and region.area_mm2 < max(4.0 * min_region_area_mm2, 28.0)
            )
            if region.max_thickness_mm + 1e-9 < required_width or awkward:
                thresholds[region.region_id] = max(
                    thresholds[region.region_id],
                    region.pixel_count + 1,
                )
                needs_merge += 1
        if needs_merge == 0:
            break

        result = merge_small_regions(
            region_labels=current_labels,
            region_palette=current_palette,
            palette_lab=palette_lab,
            min_region_pixels=min_region_pixels,
            strategy=strategy,
            color_tolerance=color_tolerance,
            region_min_pixels=thresholds,
            region_groups=_region_groups(current_labels, subject_mask),
        )
        if not result.events:
            break
        offset = len(all_events)
        all_events.extend(replace(event, step=offset + index + 1) for index, event in enumerate(result.events))
        forced_merges += result.forced_merges
        current_labels = result.region_labels
        current_palette = result.region_palette
        passes_executed += 1

    return CleanMergeResult(
        region_labels=current_labels,
        region_palette=current_palette,
        events=all_events,
        forced_merges=forced_merges,
        passes_executed=passes_executed,
    )
'''


SEGMENTATION_MODULE = r'''"""SLICO segmentation with a legacy fallback and palette-label voting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.segmentation import slic

from .color import rgb_to_lab


def _legacy_slic_labels(
    normalized_rgb: NDArray[np.uint8],
    target_segments: int,
    compactness: float,
    iterations: int = 6,
) -> NDArray[np.int32]:
    """Original local SLIC implementation kept for A/B comparisons."""
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
        counts = np.bincount(flat_labels[valid], minlength=len(centers)).astype(np.float64)
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


def _slico_labels(
    normalized_rgb: NDArray[np.uint8],
    target_segments: int,
    compactness: float,
) -> NDArray[np.int32]:
    image = normalized_rgb.astype(np.float32) / 255.0
    return slic(
        image,
        n_segments=max(10, int(target_segments)),
        compactness=float(compactness),
        sigma=0.65,
        convert2lab=True,
        enforce_connectivity=True,
        min_size_factor=0.45,
        max_size_factor=3.0,
        slic_zero=True,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)


def segment_palette_labels(
    normalized_rgb: NDArray[np.uint8],
    palette_labels: NDArray[np.int32],
    palette_size: int,
    method: str,
    superpixels: int,
    compactness: float,
    smoothing_radius: int,
) -> NDArray[np.int32]:
    """Produce a spatially coherent palette map."""
    if method not in ("components", "slic", "slic_legacy"):
        raise ValueError("segmentation doit valoir components, slic ou slic_legacy")

    if method == "components":
        segmented = palette_labels.copy()
    else:
        slic_labels = (
            _legacy_slic_labels(normalized_rgb, superpixels, compactness)
            if method == "slic_legacy"
            else _slico_labels(normalized_rgb, superpixels, compactness)
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
'''


TEST_MODULE = r'''from __future__ import annotations

import unittest

import numpy as np

from coloriage_lot3.lineart import build_line_art_mask, trace_skeleton_polylines
from coloriage_lot3.palette import merge_near_palette_colors
from coloriage_lot3.segmentation import segment_palette_labels
from coloriage_lot3.svg import region_contour_loops, shared_boundary_polylines


class CleanRenderTests(unittest.TestCase):
    def test_palette_merges_near_duplicates(self) -> None:
        centers = np.asarray(
            [[50.0, 5.0, 5.0], [50.4, 5.1, 5.0], [80.0, -10.0, 20.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
        merged, remapped, metadata = merge_near_palette_colors(
            centers,
            labels,
            threshold=3.0,
            minimum_colors=2,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(remapped.shape, labels.shape)
        self.assertEqual(metadata["merged"], 1)

    def test_slico_segmentation_preserves_shape(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, 16:] = 255
        palette = np.zeros((32, 32), dtype=np.int32)
        palette[:, 16:] = 1
        result = segment_palette_labels(
            rgb,
            palette,
            palette_size=2,
            method="slic",
            superpixels=40,
            compactness=8.0,
            smoothing_radius=0,
        )
        self.assertEqual(result.shape, palette.shape)
        self.assertGreater(np.mean(result[:, :12] == 0), 0.9)
        self.assertGreater(np.mean(result[:, 20:] == 1), 0.9)

    def test_subpixel_contours_and_shared_boundaries(self) -> None:
        labels = np.zeros((12, 12), dtype=np.uint32)
        labels[2:10, 2:6] = 1
        labels[2:10, 6:10] = 2
        loops = region_contour_loops(
            labels,
            1,
            (2, 2, 5, 9),
            smoothing_iterations=0,
            min_smooth_area_px=0.0,
            simplify_tolerance_px=0.1,
        )
        self.assertTrue(loops)
        boundaries = shared_boundary_polylines(labels)
        self.assertTrue(boundaries)
        shared_vertical = [
            path for path in boundaries
            if len(path) >= 2 and all(abs(point[0] - 6.0) < 1e-6 for point in path)
        ]
        self.assertEqual(len(shared_vertical), 1)

    def test_line_art_is_skeletonized_and_traceable(self) -> None:
        rgb = np.full((48, 48, 3), 255, dtype=np.uint8)
        rgb[10:38, 22:26] = 0
        mask, metadata = build_line_art_mask(rgb, detail_strength=0.8)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, rgb.shape[:2])
        self.assertGreaterEqual(metadata["stroke_pixels"], 1)
        self.assertTrue(trace_skeleton_polylines(mask))


if __name__ == "__main__":
    unittest.main()
'''


DOC_MODULE = r'''# Clean Render v1

Cette migration améliore rapidement la qualité du coloriage sans remplacer le
moteur déterministe par un modèle génératif.

## Changements

- prétraitement bilatéral préservant les contours ;
- SLICO (`scikit-image`) avec connectivité garantie ;
- fusion des couleurs trop proches avec Delta E 2000 ;
- passages supplémentaires de fusion pour les bandes trop fines ;
- contours de remplissage subpixel ;
- frontières partagées tracées une seule fois ;
- calque de détails internes multiscale ;
- simplification exprimée en millimètres imprimés ;
- conservation du mode historique `slic_legacy` pour les comparaisons.

## Installation

```bash
cd poc4
python -m pip install -e ".[ai]"
```

## Vérification

```bash
PYTHONPATH=poc4/src python -m unittest discover -s poc4/tests -v
npm run lint
npm test
```

## Comparaison recommandée

Générer les mêmes photos avec `slic` et `slic_legacy`, puis comparer :

- continuité des silhouettes ;
- fragmentation des visages ;
- nombre de régions après fusion ;
- nombre de polices réduites sous le seuil ;
- lisibilité sur une impression A4 réelle.
'''


@dataclass
class Change:
    path: Path
    before: str | None
    after: str


class MigrationError(RuntimeError):
    pass


def run(command: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise MigrationError(f"Ancre introuvable pour {label}")
    if count != 1:
        raise MigrationError(f"Ancre ambiguë pour {label}: {count} occurrences")
    return text.replace(old, new, 1)


def insert_before_once(text: str, anchor: str, addition: str, label: str) -> str:
    if addition.strip() in text:
        return text
    return replace_once(text, anchor, addition + anchor, label)


def replace_function(source: str, function_name: str, replacement: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        if f"def {function_name}(" in replacement and replacement.strip() in source:
            return source
        raise MigrationError(
            f"Fonction top-level {function_name!r} introuvable ou ambiguë"
        )
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    replacement = replacement.rstrip() + "\n\n"
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        if replacement.strip() in text:
            return text
        raise MigrationError(f"Début introuvable pour {label}")
    end = text.find(end_marker, start)
    if end < 0:
        raise MigrationError(f"Fin introuvable pour {label}")
    end += len(end_marker)
    return text[:start] + replacement.rstrip() + "\n" + text[end:]


def patch_pyproject(text: str) -> str:
    if '"scikit-image>=0.24,<1",' not in text:
        text = replace_once(
            text,
            '  "scipy>=1.11,<2",\n',
            '  "scipy>=1.11,<2",\n  "scikit-image>=0.24,<1",\n',
            "dépendance scikit-image",
        )
    return text


def patch_pipeline(text: str) -> str:
    if (
        "from .quality import improve_region_labelability" in text
        and "line_art_metadata=line_art_metadata" in text
    ):
        return text
    text = replace_once(
        text,
        "from .labeling import LabelPlacement, place_region_labels\n",
        "from .labeling import LabelPlacement, place_region_labels\n"
        "from .lineart import build_line_art_mask\n"
        "from .palette import merge_near_palette_colors\n"
        "from .preprocessing import prepare_processing_image\n"
        "from .quality import improve_region_labelability\n",
        "imports clean-render pipeline",
    )
    text = replace_once(
        text,
        "    min_contour_smooth_area_px: float = 18.0\n",
        "    min_contour_smooth_area_px: float = 18.0\n"
        "    contour_simplify_mm: float = 0.10\n"
        "    preprocess_sigma_color: float = 0.055\n"
        "    preprocess_sigma_spatial: float = 3.0\n"
        "    palette_merge_delta_e: float = 4.0\n"
        "    thin_merge_passes: int = 2\n"
        "    line_art_enabled: bool = True\n"
        "    line_art_detail: float = 0.65\n",
        "champs PipelineConfig clean-render",
    )
    text = replace_once(
        text,
        '        if self.segmentation not in ("components", "slic"):\n'
        '            raise ValueError("segmentation doit valoir components ou slic")\n',
        '        if self.segmentation not in ("components", "slic", "slic_legacy"):\n'
        '            raise ValueError(\n'
        '                "segmentation doit valoir components, slic ou slic_legacy"\n'
        '            )\n',
        "validation segmentation",
    )
    validation_anchor = (
        '        if self.min_contour_smooth_area_px < 0:\n'
        '            raise ValueError("min_contour_smooth_area_px doit être positif ou nul")\n'
    )
    validation_new = validation_anchor + (
        '        if not 0.0 <= self.contour_simplify_mm <= 1.0:\n'
        '            raise ValueError("contour_simplify_mm doit être compris entre 0 et 1")\n'
        '        if not 0.001 <= self.preprocess_sigma_color <= 0.25:\n'
        '            raise ValueError("preprocess_sigma_color doit être compris entre 0,001 et 0,25")\n'
        '        if not 0.25 <= self.preprocess_sigma_spatial <= 12.0:\n'
        '            raise ValueError("preprocess_sigma_spatial doit être compris entre 0,25 et 12")\n'
        '        if not 0.0 <= self.palette_merge_delta_e <= 20.0:\n'
        '            raise ValueError("palette_merge_delta_e doit être compris entre 0 et 20")\n'
        '        if not 0 <= self.thin_merge_passes <= 5:\n'
        '            raise ValueError("thin_merge_passes doit être compris entre 0 et 5")\n'
        '        if not 0.0 <= self.line_art_detail <= 1.0:\n'
        '            raise ValueError("line_art_detail doit être compris entre 0 et 1")\n'
    )
    text = replace_once(text, validation_anchor, validation_new, "validation clean-render")
    text = replace_once(
        text,
        "    detail_metadata: dict[str, Any]\n    config: PipelineConfig\n",
        "    detail_metadata: dict[str, Any]\n"
        "    line_art_mask: NDArray[np.bool_] | None\n"
        "    line_art_metadata: dict[str, Any]\n"
        "    config: PipelineConfig\n",
        "résultat line-art",
    )

    quant_start = "    start = time.perf_counter()\n    lab_image = rgb_to_lab(normalized_rgb)\n"
    quant_end = '    timings["quantization"] = (time.perf_counter() - start) * 1000.0\n'
    quant_block = r'''    start = time.perf_counter()
    processing_rgb, processing_metadata = prepare_processing_image(
        normalized_rgb,
        subject_mask=subject_mask,
        detail_mask=detail_mask,
        sigma_color=config.preprocess_sigma_color,
        sigma_spatial=config.preprocess_sigma_spatial,
    )
    source_metadata["processing"] = processing_metadata
    lab_image = rgb_to_lab(processing_rgb)
    pixels_lab = lab_image.reshape(-1, 3)
    palette_cleanup: dict[str, Any]
    if subject_mask is None:
        palette_lab, flat_labels = _fit_palette(pixels_lab, config)
        palette_lab, flat_labels, palette_cleanup = merge_near_palette_colors(
            palette_lab,
            flat_labels,
            threshold=config.palette_merge_delta_e,
            minimum_colors=2,
        )
        subject_palette_size = 0
    else:
        flat_mask = subject_mask.ravel()
        subject_requested = max(2, round(config.colors * config.subject_color_ratio))
        background_requested = max(2, config.colors - subject_requested)
        if subject_requested + background_requested > config.colors:
            subject_requested = max(2, config.colors - background_requested)
        subject_lab, subject_labels = _fit_palette(
            pixels_lab[flat_mask],
            config,
            requested_colors=subject_requested,
        )
        subject_lab, subject_labels, subject_cleanup = merge_near_palette_colors(
            subject_lab,
            subject_labels,
            threshold=config.palette_merge_delta_e,
            minimum_colors=2,
        )
        background_lab, background_labels = _fit_palette(
            pixels_lab[~flat_mask],
            config,
            requested_colors=background_requested,
        )
        background_lab, background_labels, background_cleanup = merge_near_palette_colors(
            background_lab,
            background_labels,
            threshold=config.palette_merge_delta_e,
            minimum_colors=2,
        )
        subject_palette_size = len(subject_lab)
        palette_lab = np.concatenate([subject_lab, background_lab], axis=0)
        flat_labels = np.empty(len(pixels_lab), dtype=np.int32)
        flat_labels[flat_mask] = subject_labels
        flat_labels[~flat_mask] = background_labels + subject_palette_size
        palette_cleanup = {
            "subject": subject_cleanup,
            "background": background_cleanup,
            "before": int(subject_cleanup["before"] + background_cleanup["before"]),
            "after": int(subject_cleanup["after"] + background_cleanup["after"]),
            "merged": int(subject_cleanup["merged"] + background_cleanup["merged"]),
            "threshold": float(config.palette_merge_delta_e),
        }
        subject_metadata.update(
            {
                "subject_colors": int(subject_palette_size),
                "background_colors": int(len(background_lab)),
            }
        )
    source_metadata["palette_cleanup"] = palette_cleanup
    palette_labels = flat_labels.reshape(normalized_rgb.shape[:2])
    palette_rgb = lab_to_rgb(palette_lab)
    quantized_rgb = palette_rgb[palette_labels]
    timings["quantization"] = (time.perf_counter() - start) * 1000.0
'''
    text = replace_between(text, quant_start, quant_end, quant_block, "bloc quantification")
    text = text.replace("normalized_rgb=normalized_rgb,", "normalized_rgb=processing_rgb,")

    merge_start = "    merge_result = merge_small_regions(\n"
    merge_end = "    region_palette_after = merge_result.region_palette\n"
    merge_block = r'''    merge_result = merge_small_regions(
        region_labels=region_labels_before,
        region_palette=region_palette_before,
        palette_lab=palette_lab,
        min_region_pixels=geometry.min_region_pixels,
        strategy=config.merge_strategy,
        color_tolerance=config.color_tolerance,
        region_min_pixels=region_thresholds,
        region_groups=region_groups,
    )
    region_labels_after = merge_result.region_labels
    region_palette_after = merge_result.region_palette
    clean_merge = improve_region_labelability(
        region_labels=region_labels_after,
        region_palette=region_palette_after,
        palette_lab=palette_lab,
        mm_per_pixel=geometry.mm_per_pixel,
        min_region_pixels=geometry.min_region_pixels,
        min_region_area_mm2=config.min_region_area_mm2,
        preferred_font_mm=config.number_font_mm,
        min_font_mm=config.min_number_font_mm,
        padding_mm=config.number_padding_mm,
        strategy=config.merge_strategy,
        color_tolerance=config.color_tolerance,
        subject_mask=subject_mask,
        passes=config.thin_merge_passes,
    )
    region_labels_after = clean_merge.region_labels
    region_palette_after = clean_merge.region_palette
    merge_events = [
        *merge_result.events,
        *(
            replace(event, step=len(merge_result.events) + index + 1)
            for index, event in enumerate(clean_merge.events)
        ),
    ]
    forced_merges = merge_result.forced_merges + clean_merge.forced_merges
    source_metadata["clean_merge"] = {
        "passes_executed": clean_merge.passes_executed,
        "extra_merges": len(clean_merge.events),
    }
'''
    text = replace_between(text, merge_start, merge_end, merge_block, "bloc fusion")

    line_anchor = (
        "    recolored_pixels = int(\n"
        "        np.count_nonzero(merged_palette_labels != segmented_palette_labels)\n"
        "    )\n"
        '    timings["graph_and_merge"] = (time.perf_counter() - start) * 1000.0\n'
    )
    line_new = line_anchor + r'''    line_art_mask: NDArray[np.bool_] | None = None
    line_art_metadata: dict[str, Any] = {"enabled": False}
    if config.line_art_enabled:
        line_start = time.perf_counter()
        line_art_mask, line_art_metadata = build_line_art_mask(
            processing_rgb,
            subject_mask=subject_mask,
            detail_mask=detail_mask,
            region_labels=region_labels_after,
            detail_strength=config.line_art_detail,
        )
        timings["line_art"] = (time.perf_counter() - line_start) * 1000.0
'''
    text = replace_once(text, line_anchor, line_new, "calcul line-art")
    text = replace_once(
        text,
        "        merge_events=merge_result.events,\n        forced_merges=merge_result.forced_merges,\n",
        "        merge_events=merge_events,\n        forced_merges=forced_merges,\n",
        "résultat fusion enrichie",
    )
    text = replace_once(
        text,
        "        detail_metadata=detail_metadata,\n        config=config,\n",
        "        detail_metadata=detail_metadata,\n"
        "        line_art_mask=line_art_mask,\n"
        "        line_art_metadata=line_art_metadata,\n"
        "        config=config,\n",
        "retour line-art",
    )
    return text


def patch_svg(text: str) -> str:
    if "def shared_boundary_polylines(" in text and "simplify_tolerance_px" in text:
        return text
    text = replace_once(
        text,
        "import numpy as np\n\nfrom .pipeline import PipelineResult\n",
        "import numpy as np\nfrom skimage import measure\n\n"
        "from .lineart import trace_skeleton_polylines\n"
        "from .pipeline import PipelineResult\n",
        "imports SVG clean-render",
    )
    text = replace_once(
        text,
        "    preview_supersampling: int\n",
        "    preview_supersampling: int\n    simplify_tolerance_px: float\n",
        "profil simplification SVG",
    )
    adaptive = r'''def adaptive_render_profile(result: PipelineResult) -> RenderProfile:
    """Adapte traits, lissage et simplification à la géométrie physique."""
    geometry = result.print_geometry
    page_scale = 1.12 if geometry.page_format == "a3" else 1.0
    low_resolution_boost = max(
        0.0,
        min(0.06, (geometry.mm_per_pixel - 0.16) * 0.45),
    )
    line_width_mm = max(
        0.18,
        min(
            0.42,
            result.config.line_width_mm * page_scale + low_resolution_boost,
        ),
    )
    smoothing_iterations = result.config.contour_smoothing_iterations
    if geometry.mm_per_pixel >= 0.22:
        smoothing_iterations = min(3, max(2, smoothing_iterations + 1))
    else:
        smoothing_iterations = max(1, smoothing_iterations)
    min_physical_area_mm2 = max(
        2.5,
        min(8.0, result.config.min_region_area_mm2 * 0.4),
    )
    min_smooth_area_px = max(
        result.config.min_contour_smooth_area_px,
        min_physical_area_mm2 / max(geometry.pixel_area_mm2, 1e-9),
    )
    simplify_tolerance_px = (
        result.config.contour_simplify_mm / max(geometry.mm_per_pixel, 1e-9)
    )
    return RenderProfile(
        line_width_mm=line_width_mm,
        smoothing_iterations=smoothing_iterations,
        min_smooth_area_px=min_smooth_area_px,
        preview_supersampling=2,
        simplify_tolerance_px=simplify_tolerance_px,
    )
'''
    text = replace_function(text, "adaptive_render_profile", adaptive)

    contours = r'''def region_contour_loops(
    region_labels: np.ndarray,
    region_id: int,
    bounds: tuple[int, int, int, int],
    smoothing_iterations: int = 0,
    min_smooth_area_px: float = 18.0,
    simplify_tolerance_px: float = 0.0,
) -> list[list[FloatPoint]]:
    """Return closed subpixel contours, including holes."""
    min_x, min_y, max_x, max_y = bounds
    local = (
        region_labels[min_y : max_y + 1, min_x : max_x + 1] == region_id
    )
    padded = np.pad(local.astype(np.float32), 1, mode="constant")
    contours = measure.find_contours(
        padded,
        0.5,
        fully_connected="high",
        positive_orientation="low",
    )
    loops: list[list[FloatPoint]] = []
    for contour in contours:
        if len(contour) < 4:
            continue
        # Array coordinates describe pixel centres. Subtracting 0.5 after the
        # padding maps the isoline back to physical pixel edges.
        points = np.column_stack(
            [
                contour[:, 1] - 0.5 + min_x,
                contour[:, 0] - 0.5 + min_y,
            ]
        )
        if simplify_tolerance_px > 0 and len(points) >= 6:
            points = measure.approximate_polygon(points, tolerance=simplify_tolerance_px)
        loop = [(float(x), float(y)) for x, y in points]
        if loop[0] != loop[-1]:
            loop.append(loop[0])
        if smoothing_iterations > 0 and _loop_area_px(loop) >= min_smooth_area_px:
            loop = _smooth_loop(loop[:-1], smoothing_iterations, min_smooth_area_px)
            if loop and loop[0] != loop[-1]:
                loop.append(loop[0])
        if len(loop) >= 4:
            loops.append(loop)
    return loops
'''
    text = replace_function(text, "region_contour_loops", contours)

    path_fn = r'''def region_svg_path(
    region_labels: np.ndarray,
    region_id: int,
    bounds: tuple[int, int, int, int],
    smoothing_iterations: int = 0,
    min_smooth_area_px: float = 18.0,
    simplify_tolerance_px: float = 0.0,
) -> str:
    """Vectorise a region with subpixel contours and holes."""
    loops = region_contour_loops(
        region_labels,
        region_id,
        bounds,
        smoothing_iterations=smoothing_iterations,
        min_smooth_area_px=min_smooth_area_px,
        simplify_tolerance_px=simplify_tolerance_px,
    )
    commands: list[str] = []
    for loop in loops:
        commands.append(
            f"M {_format_number(loop[0][0])} {_format_number(loop[0][1])}"
        )
        commands.extend(
            f"L {_format_number(x)} {_format_number(y)}" for x, y in loop[1:]
        )
        commands.append("Z")
    return " ".join(commands)
'''
    text = replace_function(text, "region_svg_path", path_fn)

    helpers = r'''
def _canonical_edge(start: FloatPoint, end: FloatPoint) -> tuple[FloatPoint, FloatPoint]:
    return (start, end) if start <= end else (end, start)


def _trace_unique_edges(
    edges: set[tuple[FloatPoint, FloatPoint]],
) -> list[list[FloatPoint]]:
    graph: dict[FloatPoint, set[FloatPoint]] = {}
    for start, end in edges:
        graph.setdefault(start, set()).add(end)
        graph.setdefault(end, set()).add(start)
    unused = set(edges)

    def has_edge(a: FloatPoint, b: FloatPoint) -> bool:
        return _canonical_edge(a, b) in unused

    def consume(a: FloatPoint, b: FloatPoint) -> None:
        unused.discard(_canonical_edge(a, b))

    def walk(start: FloatPoint, next_point: FloatPoint) -> list[FloatPoint]:
        path = [start, next_point]
        consume(start, next_point)
        previous = start
        current = next_point
        while True:
            candidates = [
                point
                for point in graph[current]
                if point != previous and has_edge(current, point)
            ]
            if not candidates or (len(graph[current]) != 2 and current != start):
                break
            chosen = min(candidates)
            consume(current, chosen)
            previous, current = current, chosen
            path.append(current)
            if current == start:
                break
        return path

    paths: list[list[FloatPoint]] = []
    junctions = sorted(point for point, neighbors in graph.items() if len(neighbors) != 2)
    for point in junctions:
        for neighbor in sorted(graph[point]):
            if has_edge(point, neighbor):
                paths.append(walk(point, neighbor))
    while unused:
        start, end = next(iter(unused))
        paths.append(walk(start, end))
    return paths


def shared_boundary_polylines(
    region_labels: np.ndarray,
    *,
    smoothing_iterations: int = 0,
    simplify_tolerance_px: float = 0.0,
) -> list[list[FloatPoint]]:
    """Trace every shared boundary and page edge exactly once."""
    height, width = region_labels.shape
    edges: set[tuple[FloatPoint, FloatPoint]] = set()
    for y, x in np.argwhere(region_labels[:, :-1] != region_labels[:, 1:]):
        start = (float(x + 1), float(y))
        end = (float(x + 1), float(y + 1))
        edges.add(_canonical_edge(start, end))
    for y, x in np.argwhere(region_labels[:-1, :] != region_labels[1:, :]):
        start = (float(x), float(y + 1))
        end = (float(x + 1), float(y + 1))
        edges.add(_canonical_edge(start, end))
    for x in range(width):
        edges.add(_canonical_edge((float(x), 0.0), (float(x + 1), 0.0)))
        edges.add(
            _canonical_edge(
                (float(x), float(height)),
                (float(x + 1), float(height)),
            )
        )
    for y in range(height):
        edges.add(_canonical_edge((0.0, float(y)), (0.0, float(y + 1))))
        edges.add(
            _canonical_edge(
                (float(width), float(y)),
                (float(width), float(y + 1)),
            )
        )

    output: list[list[FloatPoint]] = []
    for path in _trace_unique_edges(edges):
        points = np.asarray(path, dtype=np.float64)
        if simplify_tolerance_px > 0 and len(points) >= 4:
            points = measure.approximate_polygon(points, tolerance=simplify_tolerance_px)
        smoothed = [(float(x), float(y)) for x, y in points]
        if smoothing_iterations > 0 and len(smoothed) >= 6:
            closed = smoothed[0] == smoothed[-1]
            if closed:
                smoothed = _smooth_loop(smoothed[:-1], min(1, smoothing_iterations), 0.0)
                smoothed.append(smoothed[0])
        if len(smoothed) >= 2:
            output.append(smoothed)
    return output


def _polyline_path(points: list[FloatPoint]) -> str:
    if len(points) < 2:
        return ""
    return " ".join(
        [f"M {_format_number(points[0][0])} {_format_number(points[0][1])}"]
        + [f"L {_format_number(x)} {_format_number(y)}" for x, y in points[1:]]
    )


def _line_art_svg_paths(result: PipelineResult, tolerance_px: float) -> list[str]:
    if result.line_art_mask is None:
        return []
    paths: list[str] = []
    for polyline in trace_skeleton_polylines(result.line_art_mask):
        points = np.asarray(polyline, dtype=np.float64)
        if tolerance_px > 0 and len(points) >= 4:
            points = measure.approximate_polygon(points, tolerance=max(0.25, tolerance_px * 0.55))
        path_data = _polyline_path([(float(x), float(y)) for x, y in points])
        if path_data:
            paths.append(path_data)
    return paths

'''
    text = insert_before_once(text, "def build_svg(result: PipelineResult, colored: bool) -> str:\n", helpers, "helpers SVG")

    build_svg = r'''def build_svg(result: PipelineResult, colored: bool) -> str:
    """Construit le modèle coloré ou la feuille de coloriage numérotée."""
    geometry = result.print_geometry
    render_profile = adaptive_render_profile(result)
    stroke_px = render_profile.line_width_mm / geometry.mm_per_pixel

    fills: list[str] = []
    for region in result.regions_after:
        path_data = region_svg_path(
            result.region_labels_after,
            region.region_id,
            (region.min_x, region.min_y, region.max_x, region.max_y),
            smoothing_iterations=render_profile.smoothing_iterations,
            min_smooth_area_px=render_profile.min_smooth_area_px,
            simplify_tolerance_px=render_profile.simplify_tolerance_px,
        )
        if not path_data:
            continue
        fill = _hex(result.palette_rgb[region.palette_index]) if colored else "white"
        fills.append(
            f'<path d="{path_data}" fill="{fill}" fill-rule="evenodd" stroke="none"/>'
        )

    boundary_paths = [
        _polyline_path(path)
        for path in shared_boundary_polylines(
            result.region_labels_after,
            smoothing_iterations=render_profile.smoothing_iterations,
            simplify_tolerance_px=render_profile.simplify_tolerance_px,
        )
    ]
    boundaries = [
        f'<path d="{path}" fill="none" stroke="black" '
        f'stroke-width="{stroke_px:.5f}" stroke-linejoin="round" '
        'stroke-linecap="round"/>'
        for path in boundary_paths
        if path
    ]

    internal_details: list[str] = []
    if not colored:
        internal_width = max(0.45, stroke_px * 0.58)
        internal_details = [
            f'<path d="{path}" fill="none" stroke="#333" '
            f'stroke-width="{internal_width:.5f}" stroke-linejoin="round" '
            'stroke-linecap="round" opacity="0.82"/>'
            for path in _line_art_svg_paths(result, render_profile.simplify_tolerance_px)
        ]

    labels: list[str] = []
    if not colored:
        for placement in result.label_placements:
            if placement.status != "placed":
                continue
            font_px = placement.font_size_mm / geometry.mm_per_pixel
            halo_px = max(0.65, font_px * 0.22)
            labels.append(
                f'<text x="{placement.x_px:.3f}" y="{placement.y_px:.3f}" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="{font_px:.4f}" text-anchor="middle" '
                'dominant-baseline="central" fill="#222" paint-order="stroke" '
                f'stroke="white" stroke-width="{halo_px:.4f}" stroke-linejoin="round">'
                f'{placement.number}</text>'
            )

    subject_outline = ""
    if result.subject_mask is not None:
        mask_labels = result.subject_mask.astype(np.uint32)
        mask_path = region_svg_path(
            mask_labels,
            1,
            (0, 0, mask_labels.shape[1] - 1, mask_labels.shape[0] - 1),
            smoothing_iterations=render_profile.smoothing_iterations,
            min_smooth_area_px=render_profile.min_smooth_area_px,
            simplify_tolerance_px=render_profile.simplify_tolerance_px,
        )
        if mask_path:
            subject_outline = (
                f'<path d="{mask_path}" fill="none" fill-rule="evenodd" '
                f'stroke="black" stroke-width="{stroke_px * 1.55:.5f}" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )

    title = html.escape(result.config.title, quote=True)
    scale = geometry.mm_per_pixel
    transform = (
        f"translate({geometry.image_origin_x_mm:.6f} "
        f"{geometry.image_origin_y_mm:.6f}) scale({scale:.8f})"
    )
    page_width = geometry.page_width_mm
    page_height = geometry.page_height_mm
    mode = "Modèle coloré" if colored else "Coloriage numéroté"
    title_text = (
        ""
        if result.config.palette_layout == "separate"
        else (
            f'<text x="{page_width / 2:.3f}" y="8" '
            'font-family="Arial, Helvetica, sans-serif" font-size="4" '
            f'font-weight="bold" text-anchor="middle">{title}</text>\n'
        )
    )
    legend = (
        ""
        if result.config.palette_layout == "separate"
        else _legend_svg(result, colored=True) + "\n"
    )
    layers = [*fills, *boundaries, *internal_details, *labels]
    if subject_outline:
        layers.append(subject_outline)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width}mm" '
        f'height="{page_height}mm" viewBox="0 0 {page_width} {page_height}">\n'
        f'<title>{title} — {mode}</title>\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        + title_text
        + f'<g transform="{transform}">\n'
        + "\n".join(layers)
        + "\n</g>\n"
        + legend
        + "\n</svg>\n"
    )
'''
    text = replace_function(text, "build_svg", build_svg)
    return text


def patch_export(text: str) -> str:
    if "from .lineart import trace_skeleton_polylines" in text and "def _draw_line_art(" in text:
        return text
    text = replace_once(
        text,
        "from .pipeline import PipelineResult\n",
        "from .lineart import trace_skeleton_polylines\nfrom .pipeline import PipelineResult\n",
        "import line-art export",
    )
    text = replace_once(
        text,
        "from .svg import adaptive_render_profile, region_contour_loops, save_svgs\n",
        "from .svg import adaptive_render_profile, save_svgs, shared_boundary_polylines\n",
        "import frontières partagées export",
    )
    text = replace_once(
        text,
        '        "detail": detail_stats,\n',
        '        "detail": detail_stats,\n        "line_art": dict(result.line_art_metadata),\n',
        "stats line-art",
    )
    text = replace_once(
        text,
        '                "preview_supersampling": render_profile.preview_supersampling,\n',
        '                "preview_supersampling": render_profile.preview_supersampling,\n'
        '                "simplify_tolerance_px": render_profile.simplify_tolerance_px,\n',
        "stats simplification",
    )

    draw_contours = r'''def _draw_print_contours(
    draw: ImageDraw.ImageDraw,
    result: PipelineResult,
    pixels_per_mm: float,
) -> None:
    geometry = result.print_geometry
    render_profile = adaptive_render_profile(result)
    line_width_px = max(1, round(render_profile.line_width_mm * pixels_per_mm))
    for polyline in shared_boundary_polylines(
        result.region_labels_after,
        smoothing_iterations=render_profile.smoothing_iterations,
        simplify_tolerance_px=render_profile.simplify_tolerance_px,
    ):
        if len(polyline) < 2:
            continue
        points = [
            (
                (geometry.image_origin_x_mm + x_px * geometry.mm_per_pixel)
                * pixels_per_mm,
                (geometry.image_origin_y_mm + y_px * geometry.mm_per_pixel)
                * pixels_per_mm,
            )
            for x_px, y_px in polyline
        ]
        draw.line(points, fill="black", width=line_width_px, joint="curve")
'''
    text = replace_function(text, "_draw_print_contours", draw_contours)

    helper = r'''
def _draw_line_art(
    draw: ImageDraw.ImageDraw,
    result: PipelineResult,
    pixels_per_mm: float,
) -> None:
    if result.line_art_mask is None:
        return
    geometry = result.print_geometry
    render_profile = adaptive_render_profile(result)
    width = max(1, round(render_profile.line_width_mm * pixels_per_mm * 0.58))
    for polyline in trace_skeleton_polylines(result.line_art_mask):
        if len(polyline) < 2:
            continue
        points = [
            (
                (geometry.image_origin_x_mm + x_px * geometry.mm_per_pixel)
                * pixels_per_mm,
                (geometry.image_origin_y_mm + y_px * geometry.mm_per_pixel)
                * pixels_per_mm,
            )
            for x_px, y_px in polyline
        ]
        draw.line(points, fill="#3a3a3a", width=width, joint="curve")

'''
    text = insert_before_once(text, "def _make_print_preview(\n", helper, "dessin line-art aperçu")
    text = replace_once(
        text,
        "    _draw_print_contours(draw, result, pixels_per_mm)\n\n    if not colored:\n",
        "    _draw_print_contours(draw, result, pixels_per_mm)\n"
        "    if not colored:\n"
        "        _draw_line_art(draw, result, pixels_per_mm)\n\n"
        "    if not colored:\n",
        "appel line-art aperçu",
    )
    text = replace_once(
        text,
        '                anchor="mm",\n            )\n',
        '                anchor="mm",\n'
        '                stroke_width=max(1, round(placement.font_size_mm * pixels_per_mm * 0.18)),\n'
        '                stroke_fill="white",\n'
        '            )\n',
        "halo numéros aperçu",
    )
    return text


def patch_server(text: str) -> str:
    if "line_art_enabled=_bool_config" in text:
        return text
    bool_helper = r'''
def _clamp_float(
    value: object,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _bool_config(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "non", "off"}


'''
    text = insert_before_once(text, "def _smooth_points(\n", bool_helper, "helper bool serveur")
    text = replace_once(
        text,
        "        contour_smoothing_iterations=int(preset[\"contour_smoothing_iterations\"]),\n"
        "        title=str(payload.get(\"title\") or \"Mon coloriage mystère\"),\n",
        "        contour_smoothing_iterations=int(preset[\"contour_smoothing_iterations\"]),\n"
        "        contour_simplify_mm=_clamp_float(payload.get(\"contourSimplifyMm\"), 0.0, 0.5, 0.10),\n"
        "        preprocess_sigma_color=0.055,\n"
        "        preprocess_sigma_spatial=3.0,\n"
        "        palette_merge_delta_e=_clamp_float(payload.get(\"paletteMergeDeltaE\"), 0.0, 12.0, 4.0),\n"
        "        thin_merge_passes=_clamp_int(payload.get(\"thinMergePasses\"), 0, 5, 2),\n"
        "        line_art_enabled=_bool_config(payload.get(\"lineArtEnabled\"), True),\n"
        "        line_art_detail=_clamp_float(payload.get(\"lineArtDetail\"), 0.0, 1.0, 0.65),\n"
        "        title=str(payload.get(\"title\") or \"Mon coloriage mystère\"),\n",
        "configuration serveur clean-render",
    )
    return text


def patch_test_labeling(text: str) -> str:
    return replace_once(
        text,
        '        self.assertIn("M 0 0", path)\n',
        '        self.assertIn("M ", path)\n        self.assertIn("0.0", path)\n',
        "attente contour subpixel",
    )


def patch_cli(text: str) -> str:
    if "--no-line-art" in text and "line_art_enabled=not args.no_line_art" in text:
        return text
    text = replace_once(
        text,
        '        choices=("components", "slic"),\n',
        '        choices=("components", "slic", "slic_legacy"),\n',
        "choix segmentation CLI",
    )
    args = r'''    parser.add_argument(
        "--palette-merge-delta-e",
        type=float,
        default=4.0,
        help="Fusion perceptuelle des couleurs proches (défaut : 4,0)",
    )
    parser.add_argument(
        "--contour-simplify-mm",
        type=float,
        default=0.10,
        help="Tolérance physique de simplification vectorielle",
    )
    parser.add_argument(
        "--thin-merge-passes",
        type=int,
        default=2,
        help="Passages de fusion des zones trop étroites",
    )
    parser.add_argument(
        "--line-art-detail",
        type=float,
        default=0.65,
        help="Densité des détails internes, de 0 à 1",
    )
    parser.add_argument(
        "--no-line-art",
        action="store_true",
        help="Désactive le calque de détails internes",
    )
'''
    text = insert_before_once(
        text,
        '    parser.add_argument(\n        "--subject-mode",\n',
        args,
        "arguments CLI clean-render",
    )
    text = replace_once(
        text,
        "        line_width_mm=args.line_width_mm,\n        subject_mode=subject_mode,\n",
        "        line_width_mm=args.line_width_mm,\n"
        "        palette_merge_delta_e=args.palette_merge_delta_e,\n"
        "        contour_simplify_mm=args.contour_simplify_mm,\n"
        "        thin_merge_passes=args.thin_merge_passes,\n"
        "        line_art_enabled=not args.no_line_art,\n"
        "        line_art_detail=args.line_art_detail,\n"
        "        subject_mode=subject_mode,\n",
        "mapping CLI clean-render",
    )
    return text


def collect_changes(repo: Path) -> list[Change]:
    targets = {
        Path("poc4/pyproject.toml"): patch_pyproject,
        Path("poc4/src/coloriage_lot3/pipeline.py"): patch_pipeline,
        Path("poc4/src/coloriage_lot3/svg.py"): patch_svg,
        Path("poc4/src/coloriage_lot3/export.py"): patch_export,
        Path("poc4/src/coloriage_lot3/server.py"): patch_server,
        Path("poc4/src/coloriage_lot3/cli.py"): patch_cli,
        Path("poc4/tests/test_labeling.py"): patch_test_labeling,
    }
    changes: list[Change] = []
    for relative, patcher in targets.items():
        path = repo / relative
        if not path.is_file():
            raise MigrationError(f"Fichier attendu introuvable : {relative}")
        before = path.read_text(encoding="utf-8")
        after = patcher(before)
        changes.append(Change(path, before, after))

    generated = {
        Path("poc4/src/coloriage_lot3/preprocessing.py"): PREPROCESSING_MODULE,
        Path("poc4/src/coloriage_lot3/palette.py"): PALETTE_MODULE,
        Path("poc4/src/coloriage_lot3/lineart.py"): LINEART_MODULE,
        Path("poc4/src/coloriage_lot3/quality.py"): QUALITY_MODULE,
        Path("poc4/src/coloriage_lot3/segmentation.py"): SEGMENTATION_MODULE,
        Path("poc4/tests/test_clean_render.py"): TEST_MODULE,
        Path("poc4/CLEAN_RENDER.md"): DOC_MODULE,
    }
    for relative, content in generated.items():
        path = repo / relative
        before = path.read_text(encoding="utf-8") if path.exists() else None
        changes.append(Change(path, before, content.rstrip() + "\n"))
    return changes


def check_git_baseline(repo: Path, force: bool) -> None:
    if not (repo / ".git").exists():
        raise MigrationError("Le dossier cible n'est pas un dépôt Git")
    result = run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_COMMIT, "HEAD"],
        repo,
        check=False,
    )
    if result.returncode != 0 and not force:
        raise MigrationError(
            "Le commit de référence n'est pas un ancêtre de HEAD. "
            "Utilisez --force uniquement après avoir vérifié les différences."
        )


def backup_changes(repo: Path, changes: list[Change]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = repo / ".coloriage-migration-backup" / f"{MARKER}-{stamp}"
    for change in changes:
        if change.before is None:
            continue
        relative = change.path.relative_to(repo)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(change.before, encoding="utf-8")
    return root


def apply_changes(changes: list[Change]) -> list[Path]:
    changed: list[Path] = []
    for change in changes:
        if change.before == change.after:
            continue
        change.path.parent.mkdir(parents=True, exist_ok=True)
        change.path.write_text(change.after, encoding="utf-8")
        changed.append(change.path)
    return changed


def verify_python(repo: Path) -> None:
    package = repo / "poc4" / "src" / "coloriage_lot3"
    files = sorted(package.glob("*.py")) + sorted((repo / "poc4" / "tests").glob("*.py"))
    command = [sys.executable, "-m", "py_compile", *[str(path) for path in files]]
    result = run(command, repo, check=False)
    if result.returncode != 0:
        raise MigrationError("Compilation Python échouée :\n" + result.stdout)


def run_tests(repo: Path) -> None:
    commands = [
        [sys.executable, "-m", "pip", "install", "-e", ".[ai]"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    ]
    for command in commands:
        result = run(command, repo / "poc4", check=False)
        print(result.stdout, end="")
        if result.returncode != 0:
            raise MigrationError("Commande échouée : " + " ".join(command))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="Valide sans écrire")
    parser.add_argument("--no-tests", action="store_true", help="N'exécute pas les tests")
    parser.add_argument("--force", action="store_true", help="Ignore le contrôle d'ancêtre Git")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    try:
        check_git_baseline(repo, args.force)
        changes = collect_changes(repo)
        pending = [change for change in changes if change.before != change.after]
        print(f"Migration {MARKER}: {len(pending)} fichier(s) à modifier.")
        for change in pending:
            print(" -", change.path.relative_to(repo))
        if args.check:
            print("Vérification terminée : les ancres sont compatibles.")
            return 0
        if not pending:
            print("La migration est déjà appliquée.")
            return 0
        backup = backup_changes(repo, changes)
        changed = apply_changes(changes)
        print(f"Sauvegarde : {backup.relative_to(repo)}")
        verify_python(repo)
        if not args.no_tests:
            run_tests(repo)
        print(f"Migration appliquée sur {len(changed)} fichier(s).")
        print("Aucun commit n'a été créé. Vérifiez git diff puis committez manuellement.")
        return 0
    except (MigrationError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
