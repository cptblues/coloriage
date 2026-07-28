"""Exports diagnostics, numérotation, aperçus imprimables et SVG."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .lineart import trace_skeleton_polylines
from .pipeline import PipelineResult
from .regions import AdjacencyEdge, Region, make_region_preview
from .svg import adaptive_render_profile, save_svgs, shared_boundary_polylines
from .subject import mask_overlay


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
    placed = sum(
        placement.status == "placed" for placement in result.label_placements
    )
    skipped = len(result.label_placements) - placed
    placed_fonts = [
        placement.font_size_mm
        for placement in result.label_placements
        if placement.status == "placed"
    ]
    reduced_font_count = sum(
        font_size + 1e-9 < result.config.min_number_font_mm
        for font_size in placed_fonts
    )
    render_profile = adaptive_render_profile(result)
    subject_stats: dict[str, Any] = dict(result.subject_metadata)
    detail_stats: dict[str, Any] = dict(result.detail_metadata)
    if result.subject_mask is not None:
        ids = result.region_labels_after
        count = int(ids.max()) + 1
        totals = np.bincount(ids.ravel(), minlength=count)
        subject_pixels = np.bincount(
            ids[result.subject_mask].ravel(),
            minlength=count,
        )
        subject_regions = int(np.count_nonzero(subject_pixels[1:] * 2 >= totals[1:]))
        region_is_subject = {
            region_id: bool(subject_pixels[region_id] * 2 >= totals[region_id])
            for region_id in range(1, count)
        }
        subject_below = sum(
            region.area_mm2 < result.config.subject_min_region_area_mm2
            for region in result.regions_after
            if region_is_subject.get(region.region_id, False)
        )
        background_below = sum(
            region.area_mm2 < result.config.background_min_region_area_mm2
            for region in result.regions_after
            if not region_is_subject.get(region.region_id, False)
        )
        subject_stats.update(
            {
                "subject_regions_after": subject_regions,
                "background_regions_after": len(result.regions_after)
                - subject_regions,
                "subject_min_region_area_mm2": (
                    result.config.subject_min_region_area_mm2
                ),
                "background_min_region_area_mm2": (
                    result.config.background_min_region_area_mm2
                ),
                "subject_regions_below_threshold": int(subject_below),
                "background_regions_below_threshold": int(background_below),
            }
        )
    return {
        "schema_version": "3.4",
        "source": result.source_metadata,
        "subject": subject_stats,
        "detail": detail_stats,
        "line_art": dict(result.line_art_metadata),
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
            "labeling": {
                "region_count": len(result.label_placements),
                "placed_count": placed,
                "skipped_count": skipped,
                "coverage_percent": (
                    100.0 * placed / len(result.label_placements)
                    if result.label_placements
                    else 0.0
                ),
                "skipped_region_ids": [
                    placement.region_id
                    for placement in result.label_placements
                    if placement.status != "placed"
                ],
                "reduced_font_count": int(reduced_font_count),
                "smallest_font_mm": (
                    float(min(placed_fonts)) if placed_fonts else 0.0
                ),
            },
            "rendering": {
                "effective_line_width_mm": render_profile.line_width_mm,
                "smoothing_iterations": render_profile.smoothing_iterations,
                "min_smooth_area_px": render_profile.min_smooth_area_px,
                "preview_supersampling": render_profile.preview_supersampling,
                "simplify_tolerance_px": render_profile.simplify_tolerance_px,
            },
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
        "mean_edge_strength",
        "peak_edge_strength",
        "edge_protected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in result.merge_events:
            row = asdict(event)
            row["delta_e76"] = round(row["delta_e76"], 5)
            row["mean_edge_strength"] = round(row["mean_edge_strength"], 5)
            row["peak_edge_strength"] = round(row["peak_edge_strength"], 5)
            writer.writerow(row)


def _save_labels_csv(result: PipelineResult, path: Path) -> None:
    fields = [
        "region_id",
        "palette_index",
        "number",
        "status",
        "reason",
        "x_px",
        "y_px",
        "clearance_mm",
        "font_size_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for placement in result.label_placements:
            row = asdict(placement)
            for key in ("x_px", "y_px", "clearance_mm", "font_size_mm"):
                row[key] = round(float(row[key]), 5)
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


def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", max(1, size_px))
    except OSError:
        return ImageFont.load_default()


def _draw_print_contours(
    draw: ImageDraw.ImageDraw,
    result: PipelineResult,
    pixels_per_mm: float,
) -> None:
    geometry = result.print_geometry
    render_profile = adaptive_render_profile(result)
    line_width_px = max(1, round(render_profile.line_width_mm * pixels_per_mm))
    for polyline in shared_boundary_polylines(
        result.region_labels_after,
        smoothing_iterations=render_profile.smoothing_iterations,
        simplify_tolerance_px=render_profile.simplify_tolerance_px,
    ):
        if len(polyline) < 2:
            continue
        points = [
            (
                (geometry.image_origin_x_mm + x_px * geometry.mm_per_pixel)
                * pixels_per_mm,
                (geometry.image_origin_y_mm + y_px * geometry.mm_per_pixel)
                * pixels_per_mm,
            )
            for x_px, y_px in polyline
        ]
        draw.line(points, fill="black", width=line_width_px, joint="curve")




def _draw_line_art(
    draw: ImageDraw.ImageDraw,
    result: PipelineResult,
    pixels_per_mm: float,
) -> None:
    if result.line_art_mask is None:
        return
    geometry = result.print_geometry
    render_profile = adaptive_render_profile(result)
    width = max(1, round(render_profile.line_width_mm * pixels_per_mm * 0.58))
    for polyline in trace_skeleton_polylines(result.line_art_mask):
        if len(polyline) < 2:
            continue
        points = [
            (
                (geometry.image_origin_x_mm + x_px * geometry.mm_per_pixel)
                * pixels_per_mm,
                (geometry.image_origin_y_mm + y_px * geometry.mm_per_pixel)
                * pixels_per_mm,
            )
            for x_px, y_px in polyline
        ]
        draw.line(points, fill="#3a3a3a", width=width, joint="curve")

def _make_print_preview(
    result: PipelineResult,
    colored: bool,
    pixels_per_mm: float = 4.0,
    include_inline_palette: bool | None = None,
) -> Image.Image:
    """Rend un aperçu bitmap de la page physique sans dépendance SVG externe."""
    geometry = result.print_geometry
    output_pixels_per_mm = pixels_per_mm
    output_page_size = (
        round(geometry.page_width_mm * output_pixels_per_mm),
        round(geometry.page_height_mm * output_pixels_per_mm),
    )
    render_profile = adaptive_render_profile(result)
    supersampling = (
        render_profile.preview_supersampling
        if output_pixels_per_mm <= 6.0
        else 1
    )
    pixels_per_mm = output_pixels_per_mm * supersampling
    page_size = (
        round(geometry.page_width_mm * pixels_per_mm),
        round(geometry.page_height_mm * pixels_per_mm),
    )
    canvas = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(canvas)
    if include_inline_palette is None:
        include_inline_palette = result.config.palette_layout == "inline"
    if include_inline_palette:
        title_font = _font(round(4.0 * pixels_per_mm))
        title = result.config.title
        title_box = draw.textbbox((0, 0), title, font=title_font)
        draw.text(
            ((page_size[0] - (title_box[2] - title_box[0])) / 2, 4),
            title,
            fill="black",
            font=title_font,
        )

    image_size = (
        max(1, round(geometry.image_width_mm * pixels_per_mm)),
        max(1, round(geometry.image_height_mm * pixels_per_mm)),
    )
    image_origin = (
        round(geometry.image_origin_x_mm * pixels_per_mm),
        round(geometry.image_origin_y_mm * pixels_per_mm),
    )
    if colored:
        panel = Image.fromarray(result.merged_rgb, "RGB").resize(
            image_size,
            Image.Resampling.NEAREST,
        )
        canvas.paste(panel, image_origin)
    _draw_print_contours(draw, result, pixels_per_mm)
    if not colored:
        _draw_line_art(draw, result, pixels_per_mm)

    if not colored:
        for placement in result.label_placements:
            if placement.status != "placed":
                continue
            x_mm = (
                geometry.image_origin_x_mm
                + placement.x_px * geometry.mm_per_pixel
            )
            y_mm = (
                geometry.image_origin_y_mm
                + placement.y_px * geometry.mm_per_pixel
            )
            font = _font(round(placement.font_size_mm * pixels_per_mm))
            draw.text(
                (x_mm * pixels_per_mm, y_mm * pixels_per_mm),
                str(placement.number),
                fill="#222222",
                font=font,
                anchor="mm",
                stroke_width=max(1, round(placement.font_size_mm * pixels_per_mm * 0.18)),
                stroke_fill="white",
            )

    if include_inline_palette:
        item_width_mm = 30.0
        columns = max(1, int(geometry.printable_width_mm // item_width_mm))
        item_width_mm = geometry.printable_width_mm / columns
        legend_y = geometry.legend_origin_y_mm + 4.0
        legend_font = _font(round(2.8 * pixels_per_mm))
        draw.text(
            (geometry.margin_mm * pixels_per_mm, legend_y * pixels_per_mm),
            "Palette",
            fill="black",
            font=_font(round(3.0 * pixels_per_mm)),
            anchor="ls",
        )
        for index, rgb in enumerate(result.palette_rgb):
            row, column = divmod(index, columns)
            x_mm = geometry.margin_mm + column * item_width_mm
            y_mm = legend_y + 3.0 + row * 7.0
            xy = (
                round(x_mm * pixels_per_mm),
                round(y_mm * pixels_per_mm),
                round((x_mm + 5.0) * pixels_per_mm),
                round((y_mm + 5.0) * pixels_per_mm),
            )
            draw.rectangle(xy, fill=tuple(int(v) for v in rgb), outline="black")
            draw.text(
                ((x_mm + 7.0) * pixels_per_mm, (y_mm + 2.5) * pixels_per_mm),
                f"{index + 1} · {_rgb_hex(rgb)}",
                fill="black",
                font=legend_font,
                anchor="lm",
            )
    if supersampling > 1:
        return canvas.resize(output_page_size, Image.Resampling.LANCZOS)
    return canvas


def _make_palette_page(
    result: PipelineResult,
    pixels_per_mm: float = 4.0,
) -> Image.Image:
    """Rend la palette sur une page dédiée."""
    geometry = result.print_geometry
    page_size = (
        round(geometry.page_width_mm * pixels_per_mm),
        round(geometry.page_height_mm * pixels_per_mm),
    )
    canvas = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(round(5.0 * pixels_per_mm))
    subtitle_font = _font(round(3.0 * pixels_per_mm))
    item_font = _font(round(3.0 * pixels_per_mm))
    title = "Palette"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_y = geometry.margin_mm * pixels_per_mm
    draw.text(
        ((page_size[0] - (title_box[2] - title_box[0])) / 2, title_y),
        title,
        fill="black",
        font=title_font,
    )
    draw.text(
        (page_size[0] / 2, title_y + 7.0 * pixels_per_mm),
        result.config.title,
        fill="#555555",
        font=subtitle_font,
        anchor="mt",
    )

    item_width_mm = 42.0
    item_height_mm = 12.0
    columns = max(1, int(geometry.printable_width_mm // item_width_mm))
    item_width_mm = geometry.printable_width_mm / columns
    start_y_mm = geometry.margin_mm + 22.0
    for index, rgb in enumerate(result.palette_rgb):
        row, column = divmod(index, columns)
        x_mm = geometry.margin_mm + column * item_width_mm
        y_mm = start_y_mm + row * item_height_mm
        swatch = (
            round(x_mm * pixels_per_mm),
            round(y_mm * pixels_per_mm),
            round((x_mm + 7.0) * pixels_per_mm),
            round((y_mm + 7.0) * pixels_per_mm),
        )
        draw.rectangle(swatch, fill=tuple(int(v) for v in rgb), outline="black")
        draw.text(
            ((x_mm + 9.5) * pixels_per_mm, (y_mm + 3.5) * pixels_per_mm),
            f"{index + 1} · {_rgb_hex(rgb)}",
            fill="black",
            font=item_font,
            anchor="lm",
        )
    return canvas


def _save_pdf_document(
    result: PipelineResult,
    path: Path,
    dpi: int = 300,
) -> None:
    """Assemble le coloriage et la palette dans un PDF bitmap imprimable."""
    pixels_per_mm = dpi / 25.4
    coloring_page = _make_print_preview(
        result,
        colored=False,
        pixels_per_mm=pixels_per_mm,
        include_inline_palette=False,
    )
    palette_page = _make_palette_page(result, pixels_per_mm=pixels_per_mm)
    coloring_page.save(
        path,
        "PDF",
        resolution=dpi,
        save_all=True,
        append_images=[palette_page],
    )


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


def _make_subject_comparison(result: PipelineResult) -> Image.Image:
    if result.subject_mask is None:
        raise ValueError("Aucun masque sujet disponible")
    panels = [
        Image.fromarray(result.normalized_rgb, "RGB"),
        Image.fromarray(mask_overlay(result.normalized_rgb, result.subject_mask), "RGB"),
        Image.fromarray(
            _with_boundaries(result.merged_rgb, result.region_labels_after),
            "RGB",
        ),
    ]
    panel_width = min(420, panels[0].width)
    panel_height = max(1, round(panels[0].height * panel_width / panels[0].width))
    canvas = Image.new(
        "RGB",
        (panel_width * 3, panel_height + 42),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    titles = ["Photo", "Contrôle du masque", "Traitement sujet / fond"]
    for index, (title, panel) in enumerate(zip(titles, panels, strict=True)):
        x = index * panel_width
        draw.text((x + 10, 15), title, fill="black", font=ImageFont.load_default())
        canvas.paste(
            panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS),
            (x, 42),
        )
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
        "labels": destination / "placements-numeros.csv",
        "stats": destination / "stats.json",
        "overview": destination / "overview.png",
        "coloring_preview": destination / "apercu-coloriage.png",
        "model_print_preview": destination / "apercu-modele-couleur.png",
        "pdf_document": destination / "coloriage.pdf",
    }
    if result.config.palette_layout == "separate":
        paths["palette_page"] = destination / "palette-page.png"
    if result.subject_mask is not None:
        paths.update(
            {
                "subject_mask": destination / "masque-sujet.png",
                "mask_control": destination / "controle-masque.png",
                "subject_comparison": destination / "comparaison-sujet-fond.png",
            }
        )
    if result.detail_mask is not None:
        paths["detail_mask"] = destination / "masque-details.png"

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
    _save_labels_csv(result, paths["labels"])
    paths["stats"].write_text(
        json.dumps(build_stats(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _make_overview(result).save(paths["overview"])
    _make_print_preview(result, colored=False).save(paths["coloring_preview"])
    _make_print_preview(result, colored=True).save(paths["model_print_preview"])
    if result.config.palette_layout == "separate":
        _make_palette_page(result).save(paths["palette_page"])
    _save_pdf_document(result, paths["pdf_document"])
    if result.subject_mask is not None:
        Image.fromarray(
            (result.subject_mask.astype(np.uint8) * 255),
            "L",
        ).save(paths["subject_mask"])
        Image.fromarray(
            mask_overlay(result.normalized_rgb, result.subject_mask),
            "RGB",
        ).save(paths["mask_control"])
        _make_subject_comparison(result).save(paths["subject_comparison"])
    if result.detail_mask is not None:
        Image.fromarray(
            (result.detail_mask.astype(np.uint8) * 255),
            "L",
        ).save(paths["detail_mask"])
    paths.update(save_svgs(result, destination))
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
