from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image

from coloriage_lot3.server import (
    _draw_detail_mask,
    _resolve_max_side,
    _result_payload,
)


class ServerDetailMaskTests(unittest.TestCase):
    def test_auto_max_side_uses_source_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            small = root / "small.png"
            medium = root / "medium.png"
            large = root / "large.png"
            Image.new("RGB", (500, 300), "white").save(small)
            Image.new("RGB", (1600, 300), "white").save(medium)
            Image.new("RGB", (3000, 300), "white").save(large)

            self.assertEqual(_resolve_max_side("auto", small, 1200, 2400), 900)
            self.assertEqual(_resolve_max_side(None, medium, 1200, 2400), 1200)
            self.assertEqual(_resolve_max_side("auto", large, 1200, 2400), 1600)
            self.assertEqual(_resolve_max_side(1200, large, 1200, 2400), 1200)

    def test_outline_zone_is_filled_in_detail_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            rgb = np.full((100, 100, 3), 180, dtype=np.uint8)
            Image.fromarray(rgb, "RGB").save(source)

            path = _draw_detail_mask(
                source,
                [
                    {
                        "mode": "outline",
                        "radius": 0.02,
                        "points": [
                            {"x": 0.25, "y": 0.25},
                            {"x": 0.75, "y": 0.25},
                            {"x": 0.75, "y": 0.75},
                            {"x": 0.25, "y": 0.75},
                        ],
                    }
                ],
                root,
                200,
            )

            self.assertIsNotNone(path)
            mask = np.asarray(Image.open(path or ""))
            self.assertGreater(mask[50, 50], 0)
            self.assertEqual(mask[10, 10], 0)

    def test_result_payload_exposes_pdf_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "coloring_preview": root / "apercu-coloriage.png",
                "model_print_preview": root / "apercu-modele-couleur.png",
                "palette_page": root / "palette-page.png",
                "pdf_document": root / "coloriage.pdf",
            }
            for key, path in paths.items():
                path.write_bytes(b"%PDF" if key == "pdf_document" else b"png")

            result = SimpleNamespace(
                palette_rgb=np.asarray([[10, 20, 30]], dtype=np.uint8),
                regions_after=[object()],
            )
            payload = _result_payload(result, paths, {"result": {}})

            self.assertEqual(payload["ok"], True)
            self.assertEqual(payload["actualColors"], 1)
            self.assertTrue(
                str(payload["pdfDocument"]).startswith(
                    "data:application/pdf;base64,"
                )
            )


if __name__ == "__main__":
    unittest.main()
