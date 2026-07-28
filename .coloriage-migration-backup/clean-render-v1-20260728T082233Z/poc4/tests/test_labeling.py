from __future__ import annotations

import unittest

import numpy as np

from coloriage_lot3.geometry import compute_print_geometry
from coloriage_lot3.labeling import place_region_labels
from coloriage_lot3.regions import describe_regions
from coloriage_lot3.svg import region_svg_path


class LabelingTests(unittest.TestCase):
    def test_places_every_number_inside_including_thin_regions(self) -> None:
        labels = np.ones((20, 20), dtype=np.uint32)
        labels[:, 10:] = 2
        labels[:, 10] = 3
        palette = np.asarray([0, 0, 1, 2], dtype=np.int32)
        geometry = compute_print_geometry(
            400,
            400,
            "a4",
            "portrait",
            12.0,
            1.0,
        )
        regions = describe_regions(
            labels,
            palette,
            geometry.mm_per_pixel,
            1,
            1.5,
        )
        placements = place_region_labels(
            labels,
            palette,
            regions,
            geometry,
            preferred_font_mm=2.8,
            min_font_mm=1.8,
            padding_mm=0.45,
        )
        statuses = {item.region_id: item.status for item in placements}
        self.assertEqual(statuses[1], "placed")
        self.assertEqual(statuses[2], "placed")
        self.assertEqual(statuses[3], "placed")
        thin = next(item for item in placements if item.region_id == 3)
        self.assertGreater(thin.font_size_mm, 0.0)
        self.assertLess(thin.font_size_mm, 1.8)
        self.assertEqual(thin.reason, "police_reduite_sous_seuil")

    def test_number_size_grows_with_printed_area(self) -> None:
        labels = np.full((400, 400), 2, dtype=np.uint32)
        labels[20:26, 20:26] = 1
        palette = np.asarray([0, 0, 1], dtype=np.int32)
        geometry = compute_print_geometry(
            400,
            400,
            "a4",
            "portrait",
            12.0,
            20.0,
        )
        regions = describe_regions(
            labels,
            palette,
            geometry.mm_per_pixel,
            1,
            1.5,
        )
        placements = place_region_labels(
            labels,
            palette,
            regions,
            geometry,
            preferred_font_mm=3.2,
            min_font_mm=1.8,
            padding_mm=0.45,
        )
        by_region = {item.region_id: item for item in placements}

        self.assertEqual(by_region[1].status, "placed")
        self.assertEqual(by_region[2].status, "placed")
        self.assertLess(by_region[1].font_size_mm, by_region[2].font_size_mm)
        self.assertGreater(by_region[1].font_size_mm, 0.0)
        self.assertLessEqual(by_region[1].font_size_mm, 1.8)
        self.assertAlmostEqual(by_region[2].font_size_mm, 3.2)

    def test_svg_path_contains_closed_contours(self) -> None:
        labels = np.ones((4, 6), dtype=np.uint32)
        labels[:, 3:] = 2
        path = region_svg_path(labels, 1, (0, 0, 2, 3))
        self.assertIn("M 0 0", path)
        self.assertIn("Z", path)

    def test_svg_path_can_smooth_large_irregular_contours(self) -> None:
        labels = np.zeros((10, 10), dtype=np.uint32)
        labels[1:8, 1:4] = 1
        labels[4:8, 4:8] = 1

        path = region_svg_path(
            labels,
            1,
            (1, 1, 7, 7),
            smoothing_iterations=1,
            min_smooth_area_px=0.0,
        )

        self.assertIn(".", path)
        self.assertIn("Z", path)


if __name__ == "__main__":
    unittest.main()
