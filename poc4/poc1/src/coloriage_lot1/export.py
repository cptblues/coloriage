"""Exports des résultats du pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import PipelineResult
from .regions import make_region_preview


def _rgb_hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(value) for value in rgb))


def _region_statistics(result: PipelineResult) -> dict[str, Any]:
    sizes = np.array([region.pixel_count for region in result.regions], dtype=np.int64)
    total_pixels = int(result.region_labels.size)
    thresholds = [1, 4, 10, 25, 50, 100]
    if sizes.size == 0:
        return {
            "count": 0,
            "min_pixels": 0,
            "max_pixels": 0,
            "mean_pixels": 0.0,
            "median_pixels": 0.0,
            "regions_at_or_below_pixels": {str(value): 0 for value in thresholds},
            "pixel_share_in_regions_at_or_below": {
                str(value): 0.0 for value in thresholds
            },
        }
    return {
        "count": int(sizes.size),
        "min_pixels": int(sizes.min()),
        "max_pixels": int(sizes.max()),
        "mean_pixels": float(sizes.mean()),
        "median_pixels": float(np.median(sizes)),
        "regions_at_or_below_pixels": {
            str(value): int(np.count_nonzero(sizes <= value)) for value in thresholds
        },
        "pixel_share_in_regions_at_or_below": {
            str(value): float(100.0 * sizes[sizes <= value].sum() / total_pixels)
            for value in thresholds
        },
    }


def build_stats(result: PipelineResult) -> dict[str, Any]:
    palette_counts = np.bincount(
        result.palette_labels.ravel(),
        minlength=len(result.palette_rgb),
    )
    region_counts = np.bincount(
        [region.palette_index for region in result.regions],
        minlength=len(result.palette_rgb),
    )
    return {
        "schema_version": "1.0",
        "source": result.source_metadata,
        "parameters": asdict(result.config),
        "result": {
            "requested_colors": result.config.colors,
            "actual_colors": int(len(result.palette_rgb)),
            "pixel_count": int(result.palette_labels.size),
            "region_count": int(len(result.regions)),
            "palette_pixel_counts": [int(value) for value in palette_counts],
            "palette_region_counts": [int(value) for value in region_counts],
            "regions": _region_statistics(result),
        },
        "timings_ms": {
            key: round(float(value), 3) for key, value in result.timings_ms.items()
        },
    }


def _save_palette_csv(result: PipelineResult, path: Path) -> None:
    pixel_counts = np.bincount(
        result.palette_labels.ravel(),
        minlength=len(result.palette_rgb),
    )
    region_counts = np.bincount(
        [region.palette_index for region in result.regions],
        minlength=len(result.palette_rgb),
    )
    total = int(result.palette_labels.size)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "palette_index",
                "hex",
                "red",
                "green",
                "blue",
                "lab_l",
                "lab_a",
                "lab_b",
                "pixel_count",
                "pixel_percent",
                "region_count",
            ]
        )
        for index, (rgb, lab) in enumerate(
            zip(result.palette_rgb, result.palette_lab, strict=True)
        ):
            writer.writerow(
                [
                    index,
                    _rgb_hex(rgb),
                    *(int(value) for value in rgb),
                    *(round(float(value), 4) for value in lab),
                    int(pixel_counts[index]),
                    round(100.0 * int(pixel_counts[index]) / total, 6),
                    int(region_counts[index]),
                ]
            )


def _save_regions_csv(result: PipelineResult, path: Path) -> None:
    fields = [
        "region_id",
        "palette_index",
        "pixel_count",
        "pixel_percent",
        "min_x",
        "min_y",
        "max_x",
        "max_y",
        "width",
        "height",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for region in result.regions:
            row = asdict(region)
            row["pixel_percent"] = round(row["pixel_percent"], 8)
            writer.writerow(row)


def _make_overview(result: PipelineResult, region_preview: np.ndarray) -> Image.Image:
    source = Image.fromarray(result.normalized_rgb, "RGB")
    quantized = Image.fromarray(result.quantized_rgb, "RGB")
    regions = Image.fromarray(region_preview, "RGB")
    panel_width = min(520, source.width)
    panel_height = max(1, round(source.height * panel_width / source.width))
    source = source.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    quantized = quantized.resize((panel_width, panel_height), Image.Resampling.NEAREST)
    regions = regions.resize((panel_width, panel_height), Image.Resampling.NEAREST)

    title_height = 38
    palette_height = 94
    canvas = Image.new(
        "RGB",
        (panel_width * 3, title_height + panel_height + palette_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    titles = ["Image normalisée", "Palette réduite", "Régions connexes"]
    for index, (title, panel) in enumerate(
        zip(titles, (source, quantized, regions), strict=True)
    ):
        x = index * panel_width
        draw.text((x + 12, 13), title, fill="black", font=font)
        canvas.paste(panel, (x, title_height))

    swatch_y = title_height + panel_height + 20
    draw.text((12, swatch_y - 16), "Palette CIELAB / K-means", fill="black", font=font)
    swatch_width = max(34, min(72, (canvas.width - 24) // len(result.palette_rgb)))
    for index, rgb in enumerate(result.palette_rgb):
        x = 12 + index * swatch_width
        draw.rectangle(
            [x, swatch_y, x + swatch_width - 5, swatch_y + 36],
            fill=tuple(int(value) for value in rgb),
            outline="black",
        )
        draw.text((x, swatch_y + 43), str(index), fill="black", font=font)
    return canvas


def export_result(
    result: PipelineResult,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Le dossier de sortie n'est pas vide : {destination}. "
            "Utilisez --overwrite pour le remplacer."
        )
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "normalized": destination / "normalized.png",
        "quantized": destination / "quantized.png",
        "regions_preview": destination / "regions-preview.png",
        "region_labels": destination / "region-labels.npy",
        "palette": destination / "palette.csv",
        "regions": destination / "regions.csv",
        "stats": destination / "stats.json",
        "overview": destination / "overview.png",
    }

    Image.fromarray(result.normalized_rgb, "RGB").save(paths["normalized"])
    Image.fromarray(result.quantized_rgb, "RGB").save(paths["quantized"])
    region_preview = make_region_preview(result.region_labels)
    Image.fromarray(region_preview, "RGB").save(paths["regions_preview"])
    np.save(paths["region_labels"], result.region_labels, allow_pickle=False)
    _save_palette_csv(result, paths["palette"])
    _save_regions_csv(result, paths["regions"])
    paths["stats"].write_text(
        json.dumps(build_stats(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _make_overview(result, region_preview).save(paths["overview"])
    return paths

