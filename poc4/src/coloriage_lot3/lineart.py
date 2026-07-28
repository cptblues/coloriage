"""Multiscale internal line-art extraction and vector tracing."""

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
