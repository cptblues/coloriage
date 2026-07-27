from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from coloriage_lot3.export import export_result
from coloriage_lot3.pipeline import PipelineConfig, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_merges_and_exports_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            data = np.zeros((64, 80, 3), dtype=np.uint8)
            data[:] = [225, 70, 40]
            data[:, 40:] = [30, 100, 220]
            data[20, 20] = [30, 100, 220]
            Image.fromarray(data, "RGB").save(source)

            config = PipelineConfig(
                colors=2,
                max_side=128,
                sample_pixels=10_000,
                connectivity=8,
                seed=7,
                segmentation="components",
                smoothing_radius=0,
                min_region_area_mm2=20.0,
            )
            result = run_pipeline(source, config)
            paths = export_result(result, root / "output")

            self.assertEqual(len(result.palette_rgb), 2)
            self.assertGreater(len(result.regions_before), len(result.regions_after))
            self.assertGreaterEqual(len(result.merge_events), 1)
            for path in paths.values():
                self.assertTrue(path.is_file(), path)

            stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
            self.assertEqual(stats["schema_version"], "3.0")
            self.assertEqual(
                stats["result"]["regions_after"]["below_area_threshold_count"],
                0,
            )
            self.assertEqual(
                stats["result"]["labeling"]["placed_count"],
                len(result.regions_after),
            )

    def test_slic_path_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "gradient.png"
            x = np.linspace(0, 255, 96, dtype=np.uint8)
            data = np.tile(x, (72, 1))
            rgb = np.stack([data, np.flip(data, axis=1), data // 2], axis=-1)
            Image.fromarray(rgb, "RGB").save(source)
            result = run_pipeline(
                source,
                PipelineConfig(
                    colors=4,
                    max_side=128,
                    sample_pixels=20_000,
                    segmentation="slic",
                    superpixels=80,
                    smoothing_radius=1,
                    min_region_area_mm2=4.0,
                ),
            )
            self.assertEqual(result.region_labels_after.shape, (72, 96))
            self.assertGreater(len(result.regions_after), 0)

    def test_jpeg_input_exports_valid_svg_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "ma-photo.jpeg"
            data = np.zeros((48, 64, 3), dtype=np.uint8)
            data[:] = [235, 200, 70]
            data[:, 32:] = [40, 100, 190]
            Image.fromarray(data, "RGB").save(source, quality=92)
            result = run_pipeline(
                source,
                PipelineConfig(
                    colors=2,
                    max_side=128,
                    sample_pixels=10_000,
                    segmentation="components",
                    smoothing_radius=0,
                    min_region_area_mm2=1.0,
                ),
            )
            paths = export_result(result, root / "output")
            self.assertEqual(result.source_metadata["format"], "JPEG")
            ET.parse(paths["coloring_svg"])
            ET.parse(paths["model_svg"])
            self.assertTrue(paths["coloring_preview"].is_file())
            self.assertGreater(
                sum(item.status == "placed" for item in result.label_placements),
                0,
            )


if __name__ == "__main__":
    unittest.main()
