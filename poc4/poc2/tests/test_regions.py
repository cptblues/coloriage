from __future__ import annotations

import unittest

import numpy as np

from coloriage_lot2.regions import (
    build_adjacency,
    extract_connected_regions,
    merge_small_regions,
)


class RegionGraphTests(unittest.TestCase):
    def test_adjacency_counts_shared_boundaries(self) -> None:
        labels = np.array(
            [
                [1, 1, 2],
                [1, 3, 2],
                [3, 3, 2],
            ],
            dtype=np.uint32,
        )
        edges = {(edge.region_a, edge.region_b): edge.boundary_pixels for edge in build_adjacency(labels)}
        self.assertEqual(edges[(1, 2)], 1)
        self.assertEqual(edges[(1, 3)], 3)
        self.assertEqual(edges[(2, 3)], 2)

    def test_small_island_is_merged(self) -> None:
        palette_labels = np.zeros((7, 7), dtype=np.int32)
        palette_labels[3, 3] = 1
        region_labels, region_palette = extract_connected_regions(
            palette_labels,
            palette_size=2,
            connectivity=8,
        )
        result = merge_small_regions(
            region_labels,
            region_palette,
            np.array([[50.0, 0.0, 0.0], [55.0, 1.0, 1.0]]),
            min_region_pixels=4,
            strategy="balanced",
            color_tolerance=35.0,
        )
        self.assertEqual(int(result.region_labels.max()), 1)
        self.assertEqual(len(result.events), 1)

    def test_color_and_boundary_strategies_choose_different_neighbors(self) -> None:
        labels = np.array(
            [
                [2, 3, 3],
                [2, 1, 3],
                [3, 3, 3],
            ],
            dtype=np.uint32,
        )
        region_palette = np.array([0, 0, 1, 2], dtype=np.int32)
        palette_lab = np.array(
            [
                [50.0, 0.0, 0.0],
                [51.0, 0.0, 0.0],
                [90.0, 0.0, 0.0],
            ]
        )
        by_color = merge_small_regions(
            labels,
            region_palette,
            palette_lab,
            min_region_pixels=2,
            strategy="color",
            color_tolerance=35.0,
        )
        by_boundary = merge_small_regions(
            labels,
            region_palette,
            palette_lab,
            min_region_pixels=2,
            strategy="boundary",
            color_tolerance=35.0,
        )
        color_at_center = by_color.region_palette[by_color.region_labels[1, 1]]
        boundary_at_center = by_boundary.region_palette[
            by_boundary.region_labels[1, 1]
        ]
        self.assertEqual(int(color_at_center), 1)
        self.assertEqual(int(boundary_at_center), 2)
