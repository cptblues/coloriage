#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BRANCH_DEFAULT = "agent/ci-visual-benchmark"


def run(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd or ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def ensure_repo() -> None:
    if not (ROOT / ".git").exists() or not (ROOT / "poc4").exists():
        raise SystemExit("Place ce script à la racine du dépôt coloriage.")


def ensure_branch(branch: str) -> None:
    current = run("git", "branch", "--show-current").stdout.strip()
    if current == branch:
        return
    exists = run("git", "show-ref", "--verify", f"refs/heads/{branch}", check=False)
    run("git", "switch", branch if exists.returncode == 0 else "-c", *( [] if exists.returncode == 0 else [branch]))


def append_unique(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = block.strip()
    if block not in existing:
        path.write_text(existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def cleanup() -> None:
    backup = ROOT / ".coloriage-migration-backup"
    if backup.exists():
        shutil.rmtree(backup)
    old = ROOT / "apply_clean_render_v1.py"
    if old.exists():
        old.unlink()
    append_unique(ROOT / ".gitignore", """
# local migration and benchmark artifacts
.coloriage-migration-backup/
*.migration-backup/
benchmark-output/
poc4/benchmark/reports/
""")


def write_workflow() -> None:
    path = ROOT / ".github/workflows/quality.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('''name: quality

on:
  push:
    branches: ["main"]
  pull_request:
  workflow_dispatch:

concurrency:
  group: quality-${{ github.ref }}
  cancel-in-progress: true

jobs:
  frontend:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22.13.0"
          cache: npm
      - run: npm ci
      - run: npm run lint
      - run: npm test

  engine:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: poc4/pyproject.toml
      - run: python -m pip install -e "./poc4[ai]"
      - run: PYTHONPATH=poc4/src python -m unittest discover -s poc4/tests -v
      - run: PYTHONPATH=poc4/src python -m coloriage_lot3.benchmark --output benchmark-output --assert-quality
      - if: always()
        uses: actions/upload-artifact@v4
        with:
          name: visual-benchmark
          path: benchmark-output
          if-no-files-found: warn

  docker:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4
      - run: docker compose build
      - run: docker compose up -d
      - name: Wait for engine
        run: |
          for attempt in $(seq 1 30); do
            if curl --fail --silent http://localhost:8080/engine/health >/dev/null; then exit 0; fi
            sleep 2
          done
          docker compose logs
          exit 1
      - if: always()
        run: docker compose ps
      - if: always()
        run: docker compose down
''', encoding="utf-8")


def write_benchmark() -> None:
    path = ROOT / "poc4/src/coloriage_lot3/benchmark.py"
    path.write_text(r'''"""Deterministic visual-quality benchmark for CI and local comparisons."""
from __future__ import annotations

import argparse
import json
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
                       thin_merge_passes=0, line_art_enabled=False)
    if name == "v1":
        return replace(base, segmentation="slic")
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
    report: dict[str, Any] = {"schema_version": "1.0", "profiles": ["legacy", "v1"], "fixtures": {}}
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
            for profile_name in ("legacy", "v1"):
                started = time.perf_counter()
                result = run_pipeline(source, _profile(profile_name))
                profile_dir = fixture_dir / profile_name
                paths = export_result(result, profile_dir)
                metrics = _metrics(result, build_stats(result))
                metrics["wall_ms"] = (time.perf_counter() - started) * 1000.0
                data[profile_name] = metrics
                (profile_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                previews.append(Image.open(paths["coloring_preview"]).convert("RGB"))
            width, height = max(i.width for i in previews), max(i.height for i in previews)
            comparison = Image.new("RGB", (width * 2, height), "white")
            for index, preview in enumerate(previews):
                comparison.paste(preview, (index * width, 0))
            comparison.save(fixture_dir / "comparison.png")
            report["fixtures"][name] = data
    violations = []
    for name, data in report["fixtures"].items():
        v1 = data["v1"]
        if v1["skipped_count"] != 0: violations.append(f"{name}: numéros ignorés")
        if v1["coverage_percent"] != 100.0: violations.append(f"{name}: couverture < 100 %")
        if v1["actual_colors"] < 2: violations.append(f"{name}: palette invalide")
        if v1["regions_after"] < 1: violations.append(f"{name}: aucune région")
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
''', encoding="utf-8")


def write_test() -> None:
    path = ROOT / "poc4/tests/test_benchmark.py"
    path.write_text('''from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coloriage_lot3.benchmark import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_keeps_full_label_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark"
            report = run_benchmark(output, assert_quality=True)
            self.assertTrue((output / "report.json").is_file())
            self.assertFalse(report["violations"])
            for fixture in report["fixtures"].values():
                self.assertEqual(fixture["v1"]["skipped_count"], 0)
                self.assertEqual(fixture["v1"]["coverage_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
''', encoding="utf-8")


def write_docs() -> None:
    (ROOT / "poc4/BENCHMARK.md").write_text('''# Benchmark visuel

Le benchmark compare `slic_legacy` et le profil `v1` sur trois images synthétiques déterministes.

```bash
PYTHONPATH=poc4/src python -m coloriage_lot3.benchmark \\
  --output benchmark-output \\
  --assert-quality
```

La commande génère les aperçus, une comparaison côte à côte et un rapport JSON.
Elle échoue si un numéro est ignoré, si la couverture descend sous 100 %, ou si
la palette/région produite est invalide.
''', encoding="utf-8")


def patch_healthcheck() -> None:
    path = ROOT / "docker-compose.yml"
    content = path.read_text(encoding="utf-8")
    marker = "    volumes:\n      - rembg-models:/models/rembg\n"
    if "healthcheck:" not in content and marker in content:
        content = content.replace(marker, marker + '''    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)"]
      interval: 10s
      timeout: 5s
      retries: 6
      start_period: 20s
''')
        path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH_DEFAULT)
    parser.add_argument("--no-branch", action="store_true")
    args = parser.parse_args()
    ensure_repo()
    if not args.no_branch:
        ensure_branch(args.branch)
    cleanup()
    write_workflow()
    write_benchmark()
    write_test()
    write_docs()
    patch_healthcheck()
    for file in (ROOT / "poc4/src/coloriage_lot3/benchmark.py", ROOT / "poc4/tests/test_benchmark.py"):
        run(sys.executable, "-m", "py_compile", str(file))
    print("Migration appliquée. Vérifie avec git status puis lance les tests indiqués dans poc4/BENCHMARK.md.")


if __name__ == "__main__":
    main()
