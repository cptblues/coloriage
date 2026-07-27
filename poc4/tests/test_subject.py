from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image

from coloriage_lot3.subject import (
    generate_ai_mask,
    load_manual_mask,
    mask_overlay,
)


class SubjectMaskTests(unittest.TestCase):
    def test_manual_mask_is_resized_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mask.png"
            data = np.zeros((20, 30), dtype=np.uint8)
            data[4:18, 8:24] = 255
            Image.fromarray(data, "L").save(path)

            mask, metadata = load_manual_mask(path, (60, 40))

            self.assertEqual(mask.shape, (40, 60))
            self.assertTrue(metadata["mask_resized"])
            self.assertGreater(metadata["coverage_percent"], 10.0)
            self.assertLess(metadata["coverage_percent"], 90.0)

    def test_overlay_preserves_subject_and_tints_background(self) -> None:
        rgb = np.full((12, 12, 3), [120, 80, 40], dtype=np.uint8)
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:9, 3:9] = True

        overlay = mask_overlay(rgb, mask)

        np.testing.assert_array_equal(overlay[5, 5], rgb[5, 5])
        self.assertFalse(np.array_equal(overlay[0, 0], rgb[0, 0]))

    def test_rejects_nearly_empty_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mask.png"
            data = np.zeros((100, 100), dtype=np.uint8)
            data[0, 0] = 255
            Image.fromarray(data, "L").save(path)
            with self.assertRaisesRegex(ValueError, "moins de 0,5"):
                load_manual_mask(path, (100, 100))

    def test_ai_adapter_uses_requested_local_model(self) -> None:
        fake = ModuleType("rembg")
        calls: dict[str, object] = {}

        def new_session(model_name: str) -> object:
            calls["model"] = model_name
            return object()

        def remove(image: Image.Image, **kwargs: object) -> Image.Image:
            calls["only_mask"] = kwargs.get("only_mask")
            mask = Image.new("L", image.size, 0)
            data = np.asarray(mask).copy()
            data[4:16, 5:15] = 255
            return Image.fromarray(data, "L")

        fake.new_session = new_session  # type: ignore[attr-defined]
        fake.remove = remove  # type: ignore[attr-defined]
        rgb = np.full((20, 20, 3), 120, dtype=np.uint8)
        with patch.dict("sys.modules", {"rembg": fake}):
            mask, metadata = generate_ai_mask(rgb, "birefnet-general-lite")

        self.assertEqual(calls["model"], "birefnet-general-lite")
        self.assertTrue(calls["only_mask"])
        self.assertEqual(metadata["mode"], "ai")
        self.assertGreater(np.count_nonzero(mask), 0)


if __name__ == "__main__":
    unittest.main()
