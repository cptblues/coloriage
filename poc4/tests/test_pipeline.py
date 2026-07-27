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
            self.assertEqual(stats["schema_version"], "3.3")
            self.assertEqual(
                stats["result"]["regions_after"]["below_area_threshold_count"],
                0,
            )
            self.assertEqual(
                stats["result"]["labeling"]["placed_count"],
                len(result.regions_after),
            )
            pdf = paths["pdf_document"].read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertGreaterEqual(pdf.count(b"/Type /Page"), 2)

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

    def test_auto_tune_adjusts_generation_to_normalized_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            small_source = root / "small.png"
            large_source = root / "large.png"
            small = np.zeros((180, 180, 3), dtype=np.uint8)
            small[:, :90] = [230, 80, 50]
            small[:, 90:] = [50, 120, 220]
            large = np.zeros((180, 1500, 3), dtype=np.uint8)
            large[:, :750] = [230, 80, 50]
            large[:, 750:] = [50, 120, 220]
            Image.fromarray(small, "RGB").save(small_source)
            Image.fromarray(large, "RGB").save(large_source)
            base = dict(
                colors=2,
                max_side=1600,
                sample_pixels=4_000,
                segmentation="components",
                smoothing_radius=0,
                background_smoothing_radius=1,
                superpixels=100,
                min_region_area_mm2=10.0,
                auto_tune=True,
                contour_smoothing_iterations=0,
            )

            small_result = run_pipeline(small_source, PipelineConfig(**base))
            large_result = run_pipeline(large_source, PipelineConfig(**base))

            self.assertEqual(
                small_result.source_metadata["adaptive_profile"]["size_class"],
                "small",
            )
            self.assertEqual(
                large_result.source_metadata["adaptive_profile"]["size_class"],
                "large",
            )
            self.assertLess(small_result.config.superpixels, 100)
            self.assertGreater(large_result.config.superpixels, 100)
            self.assertGreater(small_result.config.min_region_area_mm2, 10.0)
            self.assertLess(large_result.config.min_region_area_mm2, 10.0)
            self.assertEqual(small_result.config.contour_smoothing_iterations, 2)

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

    def test_separate_palette_layout_exports_full_height_art_and_palette_page(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "portrait.png"
            data = np.zeros((120, 80, 3), dtype=np.uint8)
            data[:] = [235, 210, 80]
            data[:, 40:] = [70, 120, 210]
            Image.fromarray(data, "RGB").save(source)

            base_config = dict(
                colors=2,
                max_side=128,
                sample_pixels=10_000,
                segmentation="components",
                smoothing_radius=0,
                min_region_area_mm2=1.0,
            )
            inline = run_pipeline(source, PipelineConfig(**base_config))
            separate = run_pipeline(
                source,
                PipelineConfig(**base_config, palette_layout="separate"),
            )
            paths = export_result(separate, root / "output")

            self.assertEqual(separate.print_geometry.reserved_bottom_mm, 0.0)
            self.assertGreater(
                separate.print_geometry.image_height_mm,
                inline.print_geometry.image_height_mm,
            )
            self.assertTrue(paths["palette_page"].is_file())
            pdf = paths["pdf_document"].read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertGreaterEqual(pdf.count(b"/Type /Page"), 2)
            ET.parse(paths["palette_svg"])
            coloring_svg = paths["coloring_svg"].read_text(encoding="utf-8")
            self.assertNotIn(">Palette</text>", coloring_svg)

    def test_detail_mask_keeps_lower_region_threshold_and_numbers_match_palette(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "details.png"
            detail_path = root / "detail-mask.png"
            data = np.zeros((90, 120, 3), dtype=np.uint8)
            data[:] = [50, 130, 70]
            data[20:70, 35:85] = [215, 165, 120]
            data[34:40, 48:56] = [65, 45, 35]
            data[34:40, 66:74] = [65, 45, 35]
            Image.fromarray(data, "RGB").save(source)
            detail = np.zeros((90, 120), dtype=np.uint8)
            detail[24:56, 42:80] = 255
            Image.fromarray(detail, "L").save(detail_path)

            result = run_pipeline(
                source,
                PipelineConfig(
                    colors=6,
                    max_side=160,
                    sample_pixels=20_000,
                    segmentation="slic",
                    superpixels=120,
                    smoothing_radius=1,
                    min_region_area_mm2=20.0,
                    detail_mask_path=str(detail_path),
                    detail_min_region_area_mm2=2.0,
                ),
            )
            paths = export_result(result, root / "output")
            stats = json.loads(paths["stats"].read_text(encoding="utf-8"))

            self.assertIsNotNone(result.detail_mask)
            self.assertEqual(stats["detail"]["mode"], "manual")
            self.assertTrue(paths["detail_mask"].is_file())
            self.assertLess(
                result.config.detail_min_region_area_mm2,
                result.config.min_region_area_mm2,
            )
            for placement in result.label_placements:
                self.assertEqual(placement.number, placement.palette_index + 1)
                self.assertLessEqual(placement.number, len(result.palette_rgb))
            ET.parse(paths["model_svg"])
            self.assertTrue(paths["coloring_preview"].is_file())
            self.assertGreater(
                sum(item.status == "placed" for item in result.label_placements),
                0,
            )

    def test_manual_subject_mask_uses_split_palette_and_exports_diagnostics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "portrait.jpg"
            mask_path = root / "mask.png"
            data = np.zeros((80, 100, 3), dtype=np.uint8)
            data[:] = [55, 145, 65]
            data[12:72, 28:72] = [220, 165, 130]
            data[28:36, 38:46] = [65, 45, 35]
            data[28:36, 54:62] = [65, 45, 35]
            Image.fromarray(data, "RGB").save(source, quality=95)
            mask = np.zeros((80, 100), dtype=np.uint8)
            mask[10:74, 26:74] = 255
            Image.fromarray(mask, "L").save(mask_path)

            result = run_pipeline(
                source,
                PipelineConfig(
                    colors=8,
                    max_side=128,
                    sample_pixels=10_000,
                    segmentation="components",
                    smoothing_radius=0,
                    subject_mode="manual",
                    subject_mask_path=str(mask_path),
                    subject_color_ratio=0.625,
                    subject_min_region_area_mm2=2.0,
                    background_min_region_area_mm2=18.0,
                ),
            )
            paths = export_result(result, root / "output")
            stats = json.loads(paths["stats"].read_text(encoding="utf-8"))

            self.assertIsNotNone(result.subject_mask)
            self.assertEqual(result.subject_metadata["mode"], "manual")
            self.assertEqual(
                stats["subject"]["subject_colors"]
                + stats["subject"]["background_colors"],
                stats["result"]["actual_colors"],
            )
            self.assertGreater(stats["subject"]["subject_regions_after"], 0)
            self.assertGreater(stats["subject"]["background_regions_after"], 0)
            for key in ("subject_mask", "mask_control", "subject_comparison"):
                self.assertTrue(paths[key].is_file())
            ET.parse(paths["coloring_svg"])


if __name__ == "__main__":
    unittest.main()
