from __future__ import annotations

import unittest

import numpy as np

from coloriage_lot2.color import lab_to_rgb, rgb_to_lab


class ColorConversionTests(unittest.TestCase):
    def test_rgb_lab_round_trip(self) -> None:
        rgb = np.array(
            [[0, 0, 0], [255, 255, 255], [220, 80, 30], [25, 140, 210]],
            dtype=np.uint8,
        )
        restored = lab_to_rgb(rgb_to_lab(rgb))
        error = np.abs(restored.astype(int) - rgb.astype(int))
        self.assertLessEqual(int(error.max()), 1)


if __name__ == "__main__":
    unittest.main()
