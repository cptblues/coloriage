from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coloriage_lot1.export import export_result
from coloriage_lot1.pipeline import PipelineConfig, run_pipeline
from coloriage_lot1.regions import extract_regions


class RegionTests(unittest.TestCase):
    def test_components_are_split_by_palette_and_connectivity(self) -> None:
        labels = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
            ],
            dtype=np.int32,
        )
        region_labels, regions = extract_regions(labels, palette_size=2, connectivity=4)
        self.assertEqual(region_labels.shape, labels.shape)
        self.assertEqual(len(regions), 3)
        self.assertEqual(sum(region.pixel_count for region in regions), labels.size)


class PipelineTests(unittest.TestCase):
    def test_pipeline_exports_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            data = np.zeros((48, 64, 3), dtype=np.uint8)
            data[:, :32] = [230, 50, 30]
            data[:, 32:] = [20, 90, 220]
            Image.fromarray(data, "RGB").save(source)

            config = PipelineConfig(
                colors=2,
                max_side=128,
                sample_pixels=10_000,
                connectivity=8,
                seed=7,
            )
            result = run_pipeline(source, config)
            paths = export_result(result, root / "output")

            self.assertEqual(len(result.palette_rgb), 2)
            self.assertEqual(len(result.regions), 2)
            for path in paths.values():
                self.assertTrue(path.is_file(), path)

            stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
            self.assertEqual(stats["result"]["actual_colors"], 2)
            self.assertEqual(stats["result"]["region_count"], 2)


if __name__ == "__main__":
    unittest.main()
