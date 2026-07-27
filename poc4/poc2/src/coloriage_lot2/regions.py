"""Régions, graphe de voisinage et fusion des zones trop petites."""

from __future__ import annotations

import heapq
import math
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
    area_mm2: float
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float
    max_thickness_mm: float
    aspect_ratio: float
    compactness: float
    is_below_area_threshold: bool
    is_thin: bool


@dataclass(frozen=True)
class AdjacencyEdge:
    region_a: int
    region_b: int
    boundary_pixels: int


@dataclass(frozen=True)
class MergeEvent:
    step: int
    source_region: int
    target_region: int
    source_palette_index: int
    target_palette_index: int
    source_pixels: int
    shared_boundary_pixels: int
    delta_e76: float
    forced_by_tolerance: bool


@dataclass(frozen=True)
class MergeResult:
    region_labels: NDArray[np.uint32]
    region_palette: NDArray[np.int32]
    events: list[MergeEvent]
    forced_merges: int


def extract_connected_regions(
    palette_labels: IntArray,
    palette_size: int,
    connectivity: int = 8,
) -> tuple[NDArray[np.uint32], NDArray[np.int32]]:
    """Numérote les composantes connexes de chaque classe de palette."""
    if palette_labels.ndim != 2:
        raise ValueError("palette_labels doit être une matrice 2D")
    if connectivity not in (4, 8):
        raise ValueError("connectivity doit valoir 4 ou 8")

    structure = (
        ndimage.generate_binary_structure(2, 1)
        if connectivity == 4
        else ndimage.generate_binary_structure(2, 2)
    )
    global_labels = np.zeros(palette_labels.shape, dtype=np.uint32)
    region_palette = [0]
    next_region_id = 1

    for palette_index in range(palette_size):
        local_labels, count = ndimage.label(
            palette_labels == palette_index,
            structure=structure,
        )
        if count == 0:
            continue
        mask = local_labels > 0
        global_labels[mask] = local_labels[mask].astype(np.uint32) + next_region_id - 1
        region_palette.extend([palette_index] * int(count))
        next_region_id += int(count)

    return global_labels, np.asarray(region_palette, dtype=np.int32)


def build_adjacency(region_labels: IntArray) -> list[AdjacencyEdge]:
    """Construit le graphe de voisinage 4-connexe et mesure chaque frontière."""
    pairs: list[NDArray[np.integer]] = []
    left = region_labels[:, :-1]
    right = region_labels[:, 1:]
    mask = left != right
    if np.any(mask):
        pairs.append(np.stack([left[mask], right[mask]], axis=1))
    top = region_labels[:-1, :]
    bottom = region_labels[1:, :]
    mask = top != bottom
    if np.any(mask):
        pairs.append(np.stack([top[mask], bottom[mask]], axis=1))
    if not pairs:
        return []

    joined = np.concatenate(pairs, axis=0).astype(np.int64)
    joined.sort(axis=1)
    joined = joined[(joined[:, 0] > 0) & (joined[:, 0] != joined[:, 1])]
    if joined.size == 0:
        return []
    unique, counts = np.unique(joined, axis=0, return_counts=True)
    return [
        AdjacencyEdge(int(pair[0]), int(pair[1]), int(count))
        for pair, count in zip(unique, counts, strict=True)
    ]


def _perimeters(region_labels: IntArray) -> NDArray[np.int64]:
    count = int(region_labels.max()) + 1
    perimeter = np.zeros(count, dtype=np.int64)
    left = region_labels[:, :-1]
    right = region_labels[:, 1:]
    mask = left != right
    if np.any(mask):
        perimeter += np.bincount(left[mask], minlength=count)
        perimeter += np.bincount(right[mask], minlength=count)
    top = region_labels[:-1, :]
    bottom = region_labels[1:, :]
    mask = top != bottom
    if np.any(mask):
        perimeter += np.bincount(top[mask], minlength=count)
        perimeter += np.bincount(bottom[mask], minlength=count)
    for border in (
        region_labels[0, :],
        region_labels[-1, :],
        region_labels[:, 0],
        region_labels[:, -1],
    ):
        perimeter += np.bincount(border, minlength=count)
    return perimeter


def describe_regions(
    region_labels: IntArray,
    region_palette: IntArray,
    mm_per_pixel: float,
    min_region_pixels: int,
    thin_width_mm: float,
) -> list[Region]:
    """Calcule les métriques géométriques et physiques de chaque région."""
    if region_labels.ndim != 2:
        raise ValueError("region_labels doit être une matrice 2D")
    total_pixels = int(region_labels.size)
    counts = np.bincount(region_labels.ravel())
    perimeters = _perimeters(region_labels)
    objects = ndimage.find_objects(region_labels)
    regions: list[Region] = []

    for region_id, slices in enumerate(objects, start=1):
        if slices is None or region_id >= len(region_palette):
            continue
        ys, xs = slices
        local_mask = region_labels[ys, xs] == region_id
        pixel_count = int(counts[region_id])
        y_coords, x_coords = np.nonzero(local_mask)
        width = int(xs.stop - xs.start)
        height = int(ys.stop - ys.start)
        distance = ndimage.distance_transform_edt(local_mask)
        max_thickness_mm = float(2.0 * distance.max() * mm_per_pixel)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        perimeter = int(perimeters[region_id])
        compactness = (
            float(4.0 * math.pi * pixel_count / (perimeter**2))
            if perimeter > 0
            else 0.0
        )
        regions.append(
            Region(
                region_id=region_id,
                palette_index=int(region_palette[region_id]),
                pixel_count=pixel_count,
                pixel_percent=100.0 * pixel_count / total_pixels,
                area_mm2=float(pixel_count * mm_per_pixel**2),
                min_x=int(xs.start),
                min_y=int(ys.start),
                max_x=int(xs.stop - 1),
                max_y=int(ys.stop - 1),
                width=width,
                height=height,
                centroid_x=float(xs.start + x_coords.mean()),
                centroid_y=float(ys.start + y_coords.mean()),
                max_thickness_mm=max_thickness_mm,
                aspect_ratio=aspect_ratio,
                compactness=compactness,
                is_below_area_threshold=pixel_count < min_region_pixels,
                is_thin=max_thickness_mm < thin_width_mm,
            )
        )
    return regions


def _adjacency_dict(
    region_count: int,
    edges: list[AdjacencyEdge],
) -> dict[int, dict[int, int]]:
    graph = {region_id: {} for region_id in range(1, region_count + 1)}
    for edge in edges:
        graph[edge.region_a][edge.region_b] = edge.boundary_pixels
        graph[edge.region_b][edge.region_a] = edge.boundary_pixels
    return graph


def merge_small_regions(
    region_labels: IntArray,
    region_palette: IntArray,
    palette_lab: NDArray[np.float64],
    min_region_pixels: int,
    strategy: str,
    color_tolerance: float,
) -> MergeResult:
    """Fusionne les petites régions dans une voisine selon la stratégie choisie."""
    if strategy not in ("color", "boundary", "balanced"):
        raise ValueError("merge_strategy doit valoir color, boundary ou balanced")
    region_count = int(region_labels.max())
    if region_count == 0:
        return MergeResult(
            region_labels=np.zeros_like(region_labels, dtype=np.uint32),
            region_palette=np.zeros(1, dtype=np.int32),
            events=[],
            forced_merges=0,
        )

    edges = build_adjacency(region_labels)
    graph = _adjacency_dict(region_count, edges)
    sizes = np.bincount(region_labels.ravel(), minlength=region_count + 1).astype(
        np.int64
    )
    palettes = np.asarray(region_palette, dtype=np.int32).copy()
    parent = np.arange(region_count + 1, dtype=np.int32)
    active = np.ones(region_count + 1, dtype=bool)
    active[0] = False
    queue = [
        (int(sizes[region_id]), region_id)
        for region_id in range(1, region_count + 1)
        if sizes[region_id] < min_region_pixels
    ]
    heapq.heapify(queue)
    events: list[MergeEvent] = []
    forced_merges = 0

    while queue:
        queued_size, source = heapq.heappop(queue)
        if not active[source] or int(sizes[source]) != queued_size:
            continue
        if sizes[source] >= min_region_pixels or not graph[source]:
            continue

        candidates: list[tuple[tuple[float, ...], int, int, float]] = []
        source_palette = int(palettes[source])
        for target, boundary in graph[source].items():
            if not active[target]:
                continue
            target_palette = int(palettes[target])
            delta = float(
                np.linalg.norm(palette_lab[source_palette] - palette_lab[target_palette])
            )
            if strategy == "color":
                key = (delta, -float(boundary), -float(sizes[target]))
            elif strategy == "boundary":
                key = (-float(boundary), delta, -float(sizes[target]))
            else:
                boundary_gain = 8.0 * boundary / max(1.0, math.sqrt(sizes[source]))
                key = (delta - boundary_gain, delta, -float(boundary))
            candidates.append((key, target, boundary, delta))
        if not candidates:
            continue
        _, target, shared_boundary, delta_e = min(candidates, key=lambda item: item[0])
        forced = delta_e > color_tolerance
        forced_merges += int(forced)
        events.append(
            MergeEvent(
                step=len(events) + 1,
                source_region=source,
                target_region=target,
                source_palette_index=source_palette,
                target_palette_index=int(palettes[target]),
                source_pixels=int(sizes[source]),
                shared_boundary_pixels=int(shared_boundary),
                delta_e76=delta_e,
                forced_by_tolerance=forced,
            )
        )

        parent[source] = target
        active[source] = False
        graph[target].pop(source, None)
        for neighbor, boundary in list(graph[source].items()):
            if neighbor == target or not active[neighbor]:
                continue
            graph[neighbor].pop(source, None)
            combined = graph[target].get(neighbor, 0) + boundary
            graph[target][neighbor] = combined
            graph[neighbor][target] = combined
        graph[source].clear()
        sizes[target] += sizes[source]
        sizes[source] = 0
        if sizes[target] < min_region_pixels:
            heapq.heappush(queue, (int(sizes[target]), target))

    def root(region_id: int) -> int:
        current = region_id
        while parent[current] != current:
            current = int(parent[current])
        while parent[region_id] != region_id:
            previous = int(parent[region_id])
            parent[region_id] = current
            region_id = previous
        return current

    roots = np.zeros(region_count + 1, dtype=np.int32)
    for region_id in range(1, region_count + 1):
        roots[region_id] = root(region_id)
    active_roots = np.unique(roots[1:])
    root_to_new = np.zeros(region_count + 1, dtype=np.uint32)
    root_to_new[active_roots] = np.arange(1, len(active_roots) + 1, dtype=np.uint32)
    old_to_new = root_to_new[roots]
    merged_labels = old_to_new[region_labels].astype(np.uint32)
    merged_palette = np.zeros(len(active_roots) + 1, dtype=np.int32)
    for new_id, old_root in enumerate(active_roots, start=1):
        merged_palette[new_id] = palettes[old_root]
    return MergeResult(
        region_labels=merged_labels,
        region_palette=merged_palette,
        events=events,
        forced_merges=forced_merges,
    )


def make_region_preview(region_labels: IntArray) -> NDArray[np.uint8]:
    """Crée une visualisation déterministe aux couleurs arbitraires."""
    ids = region_labels.astype(np.uint64)
    red = (ids * 67 + 29) % 223 + 24
    green = (ids * 131 + 47) % 223 + 24
    blue = (ids * 197 + 71) % 223 + 24
    preview = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    preview[ids == 0] = 255
    return preview
