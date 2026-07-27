"""Exports des cartes de régions, graphes et métriques avant/après."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .pipeline import PipelineResult
from .regions import AdjacencyEdge, Region, make_region_preview


def _rgb_hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(value) for value in rgb))


def _region_summary(regions: list[Region]) -> dict[str, Any]:
    if not regions:
        return {
            "count": 0,
            "min_pixels": 0,
            "max_pixels": 0,
            "median_pixels": 0.0,
            "min_area_mm2": 0.0,
            "below_area_threshold_count": 0,
            "below_area_threshold_percent": 0.0,
            "thin_region_count": 0,
            "thin_region_percent": 0.0,
        }
    sizes = np.asarray([region.pixel_count for region in regions])
    areas = np.asarray([region.area_mm2 for region in regions])
    below = sum(region.is_below_area_threshold for region in regions)
    thin = sum(region.is_thin for region in regions)
    return {
        "count": len(regions),
        "min_pixels": int(sizes.min()),
        "max_pixels": int(sizes.max()),
        "median_pixels": float(np.median(sizes)),
        "min_area_mm2": float(areas.min()),
        "below_area_threshold_count": int(below),
        "below_area_threshold_percent": float(100.0 * below / len(regions)),
        "thin_region_count": int(thin),
        "thin_region_percent": float(100.0 * thin / len(regions)),
    }


def _graph_summary(regions: list[Region], edges: list[AdjacencyEdge]) -> dict[str, Any]:
    node_count = len(regions)
    return {
        "nodes": node_count,
        "edges": len(edges),
        "mean_degree": float(2.0 * len(edges) / node_count) if node_count else 0.0,
        "total_shared_boundary_pixels": int(
            sum(edge.boundary_pixels for edge in edges)
        ),
    }


def build_stats(result: PipelineResult) -> dict[str, Any]:
    total_pixels = int(result.merged_palette_labels.size)
    return {
        "schema_version": "2.0",
        "source": result.source_metadata,
        "parameters": asdict(result.config),
        "print_geometry": asdict(result.print_geometry),
        "result": {
            "requested_colors": result.config.colors,
            "actual_colors": int(len(result.palette_rgb)),
            "pixel_count": total_pixels,
            "regions_before": _region_summary(result.regions_before),
            "regions_after": _region_summary(result.regions_after),
            "graph_before": _graph_summary(
                result.regions_before,
                result.adjacency_before,
            ),
            "graph_after": _graph_summary(
                result.regions_after,
                result.adjacency_after,
            ),
            "merge_count": len(result.merge_events),
            "forced_merges_above_color_tolerance": result.forced_merges,
            "recolored_pixels": result.recolored_pixels,
            "recolored_pixel_percent": (
                100.0 * result.recolored_pixels / total_pixels
            ),
        },
        "timings_ms": {
            key: round(float(value), 3) for key, value in result.timings_ms.items()
        },
    }


def _save_palette_csv(result: PipelineResult, path: Path) -> None:
    before_counts = np.bincount(
        result.segmented_palette_labels.ravel(),
        minlength=len(result.palette_rgb),
    )
    after_counts = np.bincount(
        result.merged_palette_labels.ravel(),
        minlength=len(result.palette_rgb),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "palette_index",
                "number",
                "hex",
                "red",
                "green",
                "blue",
                "lab_l",
                "lab_a",
                "lab_b",
                "pixels_before",
                "pixels_after",
            ]
        )
        for index, (rgb, lab) in enumerate(
            zip(result.palette_rgb, result.palette_lab, strict=True)
        ):
            writer.writerow(
                [
                    index,
                    index + 1,
                    _rgb_hex(rgb),
                    *(int(value) for value in rgb),
                    *(round(float(value), 4) for value in lab),
                    int(before_counts[index]),
                    int(after_counts[index]),
                ]
            )


def _save_regions_csv(regions: list[Region], path: Path) -> None:
    fields = list(asdict(regions[0]).keys()) if regions else list(Region.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for region in regions:
            row = asdict(region)
            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = round(value, 6)
            writer.writerow(row)


def _save_adjacency_csv(
    result: PipelineResult,
    edges: list[AdjacencyEdge],
    region_palette: np.ndarray,
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "region_a",
                "region_b",
                "boundary_pixels",
                "palette_a",
                "palette_b",
                "delta_e76",
            ]
        )
        for edge in edges:
            palette_a = int(region_palette[edge.region_a])
            palette_b = int(region_palette[edge.region_b])
            delta = float(
                np.linalg.norm(
                    result.palette_lab[palette_a] - result.palette_lab[palette_b]
                )
            )
            writer.writerow(
                [
                    edge.region_a,
                    edge.region_b,
                    edge.boundary_pixels,
                    palette_a,
                    palette_b,
                    round(delta, 5),
                ]
            )


def _save_merges_csv(result: PipelineResult, path: Path) -> None:
    fields = [
        "step",
        "source_region",
        "target_region",
        "source_palette_index",
        "target_palette_index",
        "source_pixels",
        "shared_boundary_pixels",
        "delta_e76",
        "forced_by_tolerance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in result.merge_events:
            row = asdict(event)
            row["delta_e76"] = round(row["delta_e76"], 5)
            writer.writerow(row)


def _boundary_mask(region_labels: np.ndarray, line_width: int = 1) -> np.ndarray:
    boundary = np.zeros(region_labels.shape, dtype=bool)
    differences = region_labels[:, :-1] != region_labels[:, 1:]
    boundary[:, :-1] |= differences
    boundary[:, 1:] |= differences
    differences = region_labels[:-1, :] != region_labels[1:, :]
    boundary[:-1, :] |= differences
    boundary[1:, :] |= differences
    if line_width > 1:
        boundary = ndimage.maximum_filter(
            boundary,
            size=2 * line_width - 1,
            mode="nearest",
        )
    return boundary


def _with_boundaries(
    rgb: np.ndarray,
    region_labels: np.ndarray,
    line_width: int = 1,
) -> np.ndarray:
    output = rgb.copy()
    output[_boundary_mask(region_labels, line_width)] = 0
    return output


def _contours_only(region_labels: np.ndarray, line_width: int = 1) -> np.ndarray:
    output = np.full((*region_labels.shape, 3), 255, dtype=np.uint8)
    output[_boundary_mask(region_labels, line_width)] = 0
    return output


def _make_overview(result: PipelineResult) -> Image.Image:
    panels = [
        Image.fromarray(result.normalized_rgb, "RGB"),
        Image.fromarray(result.quantized_rgb, "RGB"),
        Image.fromarray(
            _with_boundaries(result.segmented_rgb, result.region_labels_before),
            "RGB",
        ),
        Image.fromarray(
            _with_boundaries(result.merged_rgb, result.region_labels_after),
            "RGB",
        ),
    ]
    panel_width = min(440, panels[0].width)
    panel_height = max(1, round(panels[0].height * panel_width / panels[0].width))
    panels = [
        panel.resize(
            (panel_width, panel_height),
            Image.Resampling.LANCZOS if index < 2 else Image.Resampling.NEAREST,
        )
        for index, panel in enumerate(panels)
    ]
    title_height = 44
    footer_height = 104
    canvas = Image.new(
        "RGB",
        (panel_width * 2, (title_height + panel_height) * 2 + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    titles = [
        "Image normalisée",
        "Palette réduite",
        f"Avant fusion — {len(result.regions_before)} régions",
        f"Après fusion — {len(result.regions_after)} régions",
    ]
    for index, (title, panel) in enumerate(zip(titles, panels, strict=True)):
        column = index % 2
        row = index // 2
        x = column * panel_width
        y = row * (title_height + panel_height)
        draw.text((x + 12, y + 15), title, fill="black", font=font)
        canvas.paste(panel, (x, y + title_height))

    footer_y = 2 * (title_height + panel_height) + 12
    stats = build_stats(result)
    after = stats["result"]["regions_after"]
    lines = [
        (
            f"Seuil: {result.print_geometry.min_region_area_mm2:.1f} mm² "
            f"= {result.print_geometry.min_region_pixels} px "
            f"({result.config.page_format.upper()} {result.config.orientation})"
        ),
        (
            f"Fusions: {len(result.merge_events)} | "
            f"zones sous seuil restantes: {after['below_area_threshold_count']} | "
            f"zones fines: {after['thin_region_count']}"
        ),
        (
            f"Pixels recolorés: {stats['result']['recolored_pixel_percent']:.3f}% | "
            f"durée: {result.timings_ms['total']:.0f} ms | "
            f"stratégie: {result.config.merge_strategy}"
        ),
    ]
    for offset, line in enumerate(lines):
        draw.text((12, footer_y + offset * 22), line, fill="black", font=font)
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
        "segmented": destination / "segmented-before.png",
        "merged": destination / "merged-after.png",
        "regions_before_preview": destination / "regions-before-preview.png",
        "regions_after_preview": destination / "regions-after-preview.png",
        "contours_before": destination / "contours-before.png",
        "contours_after": destination / "contours-after.png",
        "model_before": destination / "model-before.png",
        "model_after": destination / "model-after.png",
        "labels_before": destination / "region-labels-before.npy",
        "labels_after": destination / "region-labels-after.npy",
        "palette": destination / "palette.csv",
        "regions_before": destination / "regions-before.csv",
        "regions_after": destination / "regions-after.csv",
        "adjacency_before": destination / "adjacency-before.csv",
        "adjacency_after": destination / "adjacency-after.csv",
        "merges": destination / "merges.csv",
        "stats": destination / "stats.json",
        "overview": destination / "overview.png",
    }

    Image.fromarray(result.normalized_rgb, "RGB").save(paths["normalized"])
    Image.fromarray(result.quantized_rgb, "RGB").save(paths["quantized"])
    Image.fromarray(result.segmented_rgb, "RGB").save(paths["segmented"])
    Image.fromarray(result.merged_rgb, "RGB").save(paths["merged"])
    Image.fromarray(
        make_region_preview(result.region_labels_before),
        "RGB",
    ).save(paths["regions_before_preview"])
    Image.fromarray(
        make_region_preview(result.region_labels_after),
        "RGB",
    ).save(paths["regions_after_preview"])
    Image.fromarray(_contours_only(result.region_labels_before), "RGB").save(
        paths["contours_before"]
    )
    Image.fromarray(_contours_only(result.region_labels_after), "RGB").save(
        paths["contours_after"]
    )
    Image.fromarray(
        _with_boundaries(result.segmented_rgb, result.region_labels_before),
        "RGB",
    ).save(paths["model_before"])
    Image.fromarray(
        _with_boundaries(result.merged_rgb, result.region_labels_after),
        "RGB",
    ).save(paths["model_after"])
    np.save(paths["labels_before"], result.region_labels_before, allow_pickle=False)
    np.save(paths["labels_after"], result.region_labels_after, allow_pickle=False)
    _save_palette_csv(result, paths["palette"])
    _save_regions_csv(result.regions_before, paths["regions_before"])
    _save_regions_csv(result.regions_after, paths["regions_after"])
    _save_adjacency_csv(
        result,
        result.adjacency_before,
        result.region_palette_before,
        paths["adjacency_before"],
    )
    _save_adjacency_csv(
        result,
        result.adjacency_after,
        result.region_palette_after,
        paths["adjacency_after"],
    )
    _save_merges_csv(result, paths["merges"])
    paths["stats"].write_text(
        json.dumps(build_stats(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _make_overview(result).save(paths["overview"])
    return paths


def export_strategy_comparison(
    results: dict[str, PipelineResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Exporte un tableau et une planche comparant plusieurs règles de fusion."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "strategies-comparison.csv"
    json_path = destination / "strategies-comparison.json"
    image_path = destination / "strategies-comparison.png"
    rows: list[dict[str, Any]] = []
    panels: list[tuple[str, PipelineResult, Image.Image]] = []
    for name, result in results.items():
        stats = build_stats(result)
        after = stats["result"]["regions_after"]
        row = {
            "strategy": name,
            "regions_before": len(result.regions_before),
            "regions_after": len(result.regions_after),
            "merges": len(result.merge_events),
            "forced_merges": result.forced_merges,
            "below_threshold_after": after["below_area_threshold_count"],
            "thin_regions_after": after["thin_region_count"],
            "recolored_pixel_percent": round(
                stats["result"]["recolored_pixel_percent"],
                6,
            ),
            "duration_ms": round(result.timings_ms["total"], 3),
        }
        rows.append(row)
        panels.append(
            (
                name,
                result,
                Image.fromarray(
                    _with_boundaries(result.merged_rgb, result.region_labels_after),
                    "RGB",
                ),
            )
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    panel_width = min(420, panels[0][2].width)
    panel_height = round(panels[0][2].height * panel_width / panels[0][2].width)
    title_height = 64
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), title_height + panel_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (name, result, panel) in enumerate(panels):
        x = index * panel_width
        resized = panel.resize((panel_width, panel_height), Image.Resampling.NEAREST)
        row = rows[index]
        draw.text((x + 10, 12), name, fill="black", font=font)
        draw.text(
            (x + 10, 32),
            (
                f"{row['regions_after']} régions | "
                f"{row['recolored_pixel_percent']:.3f}% recoloré"
            ),
            fill="black",
            font=font,
        )
        canvas.paste(resized, (x, title_height))
    canvas.save(image_path)
    return {"csv": csv_path, "json": json_path, "image": image_path}
