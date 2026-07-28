"""Deterministic visual-quality benchmark for CI and local comparisons."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .export import build_stats, export_result
from .pipeline import PipelineConfig, run_pipeline


def _portrait(path: Path) -> None:
    image = Image.new("RGB", (180, 220), (224, 230, 216))
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 24, 138, 128), fill=(222, 171, 137))
    draw.pieslice((34, 12, 146, 112), 180, 360, fill=(75, 47, 35))
    draw.ellipse((68, 62, 80, 72), fill=(45, 35, 30))
    draw.ellipse((100, 62, 112, 72), fill=(45, 35, 30))
    draw.arc((76, 73, 105, 102), 15, 165, fill=(135, 72, 65), width=3)
    draw.polygon([(52, 118), (128, 118), (154, 212), (26, 212)], fill=(62, 111, 168))
    image.save(path)


def _pet(path: Path) -> None:
    image = Image.new("RGB", (220, 170), (235, 226, 204))
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 28, 174, 145), fill=(170, 127, 82))
    draw.polygon([(62, 48), (54, 4), (94, 36)], fill=(125, 87, 58))
    draw.polygon([(154, 42), (181, 5), (184, 66)], fill=(125, 87, 58))
    draw.ellipse((84, 70, 98, 84), fill=(35, 30, 25))
    draw.ellipse((128, 70, 142, 84), fill=(35, 30, 25))
    draw.ellipse((103, 91, 124, 108), fill=(65, 49, 40))
    for y in range(40, 135, 12):
        draw.arc((55, y, 170, y + 25), 195, 345, fill=(151, 107, 73), width=2)
    image.save(path)


def _object(path: Path) -> None:
    image = Image.new("RGB", (240, 170), (246, 244, 236))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((42, 30, 198, 142), radius=18, fill=(68, 124, 180))
    draw.rectangle((64, 50, 176, 122), fill=(230, 191, 82))
    draw.ellipse((85, 64, 155, 120), fill=(201, 72, 61))
    image.save(path)


FIXTURES = {"portrait": _portrait, "pet": _pet, "object": _object}


def _profile(name: str) -> PipelineConfig:
    base = PipelineConfig(colors=8, max_side=320, sample_pixels=30000, superpixels=180,
                          min_region_area_mm2=6.0, palette_layout="separate",
                          auto_tune=False, line_art_enabled=True)
    if name == "legacy":
        return replace(base, segmentation="slic_legacy", palette_merge_delta_e=0.0,
                       palette_mode="legacy", edge_guided_merge=False,
                       thin_merge_passes=0, line_art_enabled=False)
    if name == "v1":
        return replace(
            base, segmentation="slic", palette_mode="legacy",
            edge_guided_merge=False,
        )
    if name == "v2":
        return replace(
            base, segmentation="slic", palette_mode="adaptive",
            edge_guided_merge=True,
        )
    raise ValueError(name)


def _metrics(result: Any, stats: dict[str, Any]) -> dict[str, Any]:
    labeling = stats["result"]["labeling"]
    regions = stats["result"]["regions_after"]
    return {
        "actual_colors": int(len(result.palette_rgb)),
        "regions_after": int(len(result.regions_after)),
        "thin_regions": int(regions["thin_region_count"]),
        "skipped_count": int(labeling["skipped_count"]),
        "coverage_percent": float(labeling["coverage_percent"]),
        "reduced_font_count": int(labeling["reduced_font_count"]),
        "smallest_font_mm": float(labeling["smallest_font_mm"]),
        "forced_merges": int(result.forced_merges),
        "recolored_pixels": int(result.recolored_pixels),
        "line_art_pixels": int(np.count_nonzero(result.line_art_mask)) if result.line_art_mask is not None else 0,
        "total_ms": float(result.timings_ms.get("total", 0.0)),
    }


def run_benchmark(output: Path, assert_quality: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": "1.1", "profiles": ["legacy", "v1", "v2"], "fixtures": {}}
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for name, builder in FIXTURES.items():
            source = root / f"{name}.png"
            builder(source)
            fixture_dir = output / name
            fixture_dir.mkdir(exist_ok=True)
            (fixture_dir / "source.png").write_bytes(source.read_bytes())
            previews = []
            data = {}
            for profile_name in ("legacy", "v1", "v2"):
                started = time.perf_counter()
                result = run_pipeline(source, _profile(profile_name))
                profile_dir = fixture_dir / profile_name
                if profile_dir.exists():
                    shutil.rmtree(profile_dir)
                paths = export_result(result, profile_dir)
                metrics = _metrics(result, build_stats(result))
                metrics["wall_ms"] = (time.perf_counter() - started) * 1000.0
                data[profile_name] = metrics
                (profile_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                previews.append(Image.open(paths["coloring_preview"]).convert("RGB"))
            width, height = max(i.width for i in previews), max(i.height for i in previews)
            comparison = Image.new("RGB", (width * len(previews), height), "white")
            for index, preview in enumerate(previews):
                comparison.paste(preview, (index * width, 0))
            comparison.save(fixture_dir / "comparison.png")
            report["fixtures"][name] = data
    violations = []
    for name, data in report["fixtures"].items():
        v2 = data["v2"]
        if v2["skipped_count"] != 0: violations.append(f"{name}: numéros ignorés")
        if v2["coverage_percent"] != 100.0: violations.append(f"{name}: couverture < 100 %")
        if v2["actual_colors"] < 2: violations.append(f"{name}: palette invalide")
        if v2["regions_after"] < 1: violations.append(f"{name}: aucune région")
    report["violations"] = violations
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if assert_quality and violations:
        raise SystemExit("Benchmark échoué:\n- " + "\n- ".join(violations))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--assert-quality", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output, args.assert_quality), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
