from __future__ import annotations

import unittest

from coloriage_lot2.geometry import compute_print_geometry


class PrintGeometryTests(unittest.TestCase):
    def test_a4_needs_more_pixels_than_a3_for_same_physical_area(self) -> None:
        a4 = compute_print_geometry(960, 720, "a4", "portrait", 12.0, 9.0)
        a3 = compute_print_geometry(960, 720, "a3", "portrait", 12.0, 9.0)
        self.assertGreater(a4.min_region_pixels, a3.min_region_pixels)
        self.assertAlmostEqual(a4.min_region_area_mm2, 9.0)

    def test_landscape_changes_available_scale(self) -> None:
        portrait = compute_print_geometry(1200, 600, "a4", "portrait", 12.0, 9.0)
        landscape = compute_print_geometry(1200, 600, "a4", "landscape", 12.0, 9.0)
        self.assertGreater(landscape.mm_per_pixel, portrait.mm_per_pixel)
