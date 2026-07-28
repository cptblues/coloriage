from __future__ import annotations

import unittest

import numpy as np

from coloriage_lot3.edges import build_edge_strength_map, measure_region_boundaries
from coloriage_lot3.palette import build_global_palette
from coloriage_lot3.regions import merge_small_regions


class PaletteEdgeV2Tests(unittest.TestCase):
    def test_global_palette_merges_subject_background_duplicates(self) -> None:
        pixels = np.asarray(
            [[50.0, 5.0, 5.0], [50.2, 5.1, 5.0], [80.0, -8.0, 16.0], [20.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        centers = pixels.copy()
        labels = np.arange(4, dtype=np.int32)
        result = build_global_palette(
            pixels,
            centers,
            labels,
            requested_colors=4,
            mode="adaptive",
            merge_threshold=2.5,
        )
        self.assertEqual(len(result.centers_lab), 3)
        self.assertEqual(result.labels[0], result.labels[1])

    def test_exact_palette_replenishes_requested_colors(self) -> None:
        pixels = np.asarray(
            [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0], [60.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        centers = np.asarray([[20.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float64)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        result = build_global_palette(
            pixels,
            centers,
            labels,
            requested_colors=4,
            mode="exact",
            merge_threshold=0.0,
        )
        self.assertEqual(len(result.centers_lab), 4)
        self.assertTrue(result.metadata["exact_achieved"])

    def test_edge_guidance_avoids_strong_boundary(self) -> None:
        labels = np.asarray(
            [
                [1, 1, 2, 3, 3],
                [1, 1, 2, 3, 3],
                [1, 1, 2, 3, 3],
            ],
            dtype=np.uint32,
        )
        region_palette = np.asarray([0, 0, 1, 2], dtype=np.int32)
        palette = np.asarray(
            [[50.0, 0.0, 0.0], [50.5, 0.0, 0.0], [57.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        edge_map = np.zeros(labels.shape, dtype=np.float32)
        edge_map[:, 1] = 1.0
        result = merge_small_regions(
            region_labels=labels,
            region_palette=region_palette,
            palette_lab=palette,
            min_region_pixels=4,
            strategy="balanced",
            color_tolerance=35.0,
            edge_strength_map=edge_map,
            edge_weight=30.0,
            edge_protection_threshold=0.7,
        )
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].target_region, 3)
        self.assertFalse(result.events[0].edge_protected)

    def test_semantic_subject_outline_is_protected(self) -> None:
        rgb = np.full((32, 32, 3), 220, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        strength, metadata = build_edge_strength_map(rgb, subject_mask=mask)
        self.assertGreaterEqual(float(strength[8, 16]), 0.9)
        self.assertGreater(metadata["semantic_protected_pixels"], 0)
        regions = np.where(mask, 1, 2).astype(np.uint32)
        guidance = measure_region_boundaries(regions, strength)
        self.assertTrue(guidance)
        self.assertGreater(max(item.mean_strength for item in guidance.values()), 0.7)


if __name__ == "__main__":
    unittest.main()
