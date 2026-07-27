from __future__ import annotations

import unittest

import numpy as np

from coloriage_lot3.geometry import compute_print_geometry
from coloriage_lot3.labeling import place_region_labels
from coloriage_lot3.regions import describe_regions
from coloriage_lot3.svg import region_svg_path


class LabelingTests(unittest.TestCase):
    def test_places_number_in_large_region_and_skips_thin_region(self) -> None:
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
        self.assertEqual(statuses[3], "skipped")

    def test_svg_path_contains_closed_contours(self) -> None:
        labels = np.ones((4, 6), dtype=np.uint32)
        labels[:, 3:] = 2
        path = region_svg_path(labels, 1, (0, 0, 2, 3))
        self.assertIn("M 0 0", path)
        self.assertIn("Z", path)


if __name__ == "__main__":
    unittest.main()
