from __future__ import annotations

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
