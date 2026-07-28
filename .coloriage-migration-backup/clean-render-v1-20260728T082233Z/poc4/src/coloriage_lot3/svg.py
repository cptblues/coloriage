"""Génération de feuilles vectorielles SVG prêtes à imprimer."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import PipelineResult

Point = tuple[int, int]
FloatPoint = tuple[float, float]
Edge = tuple[Point, Point]


@dataclass(frozen=True)
class RenderProfile:
    """Paramètres de tracé dérivés de la taille imprimée et de la résolution."""

    line_width_mm: float
    smoothing_iterations: int
    min_smooth_area_px: float
    preview_supersampling: int


def adaptive_render_profile(result: PipelineResult) -> RenderProfile:
    """Adapte traits et lissage à la géométrie physique du document."""
    geometry = result.print_geometry
    page_scale = 1.12 if geometry.page_format == "a3" else 1.0
    low_resolution_boost = max(
        0.0,
        min(0.06, (geometry.mm_per_pixel - 0.16) * 0.45),
    )
    line_width_mm = max(
        0.18,
        min(
            0.42,
            result.config.line_width_mm * page_scale + low_resolution_boost,
        ),
    )

    smoothing_iterations = result.config.contour_smoothing_iterations
    if geometry.mm_per_pixel >= 0.22:
        smoothing_iterations = min(3, max(2, smoothing_iterations + 1))
    else:
        smoothing_iterations = max(1, smoothing_iterations)

    min_physical_area_mm2 = max(
        2.5,
        min(8.0, result.config.min_region_area_mm2 * 0.4),
    )
    min_smooth_area_px = max(
        result.config.min_contour_smooth_area_px,
        min_physical_area_mm2 / max(geometry.pixel_area_mm2, 1e-9),
    )
    return RenderProfile(
        line_width_mm=line_width_mm,
        smoothing_iterations=smoothing_iterations,
        min_smooth_area_px=min_smooth_area_px,
        preview_supersampling=2,
    )


def _direction(start: Point, end: Point) -> int:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(dx, dy)]


def _mask_edges(mask: np.ndarray, x_offset: int, y_offset: int) -> set[Edge]:
    above = np.zeros_like(mask)
    above[1:] = mask[:-1]
    below = np.zeros_like(mask)
    below[:-1] = mask[1:]
    left = np.zeros_like(mask)
    left[:, 1:] = mask[:, :-1]
    right = np.zeros_like(mask)
    right[:, :-1] = mask[:, 1:]
    edges: set[Edge] = set()

    for y, x in np.argwhere(mask & ~above):
        edges.add(
            ((x_offset + int(x), y_offset + int(y)),
             (x_offset + int(x) + 1, y_offset + int(y)))
        )
    for y, x in np.argwhere(mask & ~right):
        edges.add(
            ((x_offset + int(x) + 1, y_offset + int(y)),
             (x_offset + int(x) + 1, y_offset + int(y) + 1))
        )
    for y, x in np.argwhere(mask & ~below):
        edges.add(
            ((x_offset + int(x) + 1, y_offset + int(y) + 1),
             (x_offset + int(x), y_offset + int(y) + 1))
        )
    for y, x in np.argwhere(mask & ~left):
        edges.add(
            ((x_offset + int(x), y_offset + int(y) + 1),
             (x_offset + int(x), y_offset + int(y)))
        )
    return edges


def _trace_edges(edges: set[Edge]) -> list[list[Point]]:
    by_start: dict[Point, list[Point]] = {}
    for start, end in edges:
        by_start.setdefault(start, []).append(end)
    unused = set(edges)
    loops: list[list[Point]] = []

    while unused:
        start, first_end = min(unused)
        unused.remove((start, first_end))
        raw = [start, first_end]
        previous_direction = _direction(start, first_end)
        current = first_end
        guard = 0
        while current != start and guard <= len(edges):
            guard += 1
            candidates = [
                end
                for end in by_start.get(current, [])
                if (current, end) in unused
            ]
            if not candidates:
                break
            turn_rank = {1: 0, 0: 1, 3: 2, 2: 3}
            chosen = min(
                candidates,
                key=lambda end: (
                    turn_rank[
                        (_direction(current, end) - previous_direction) % 4
                    ],
                    end,
                ),
            )
            unused.remove((current, chosen))
            previous_direction = _direction(current, chosen)
            current = chosen
            raw.append(current)

        if raw[-1] != start or len(raw) < 5:
            continue
        compact = [raw[0]]
        for index in range(1, len(raw) - 1):
            before = raw[index - 1]
            point = raw[index]
            after = raw[index + 1]
            if _direction(before, point) != _direction(point, after):
                compact.append(point)
        if len(compact) >= 3:
            loops.append(compact)
    return loops


def _loop_area_px(loop: list[FloatPoint]) -> float:
    if len(loop) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(loop):
        next_point = loop[(index + 1) % len(loop)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def _smooth_loop(
    loop: list[Point],
    iterations: int,
    min_area_px: float,
) -> list[FloatPoint]:
    points = [(float(x), float(y)) for x, y in loop]
    if iterations <= 0 or len(points) < 6 or _loop_area_px(points) < min_area_px:
        return points

    for _ in range(iterations):
        smoothed: list[FloatPoint] = []
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            smoothed.append(
                (
                    point[0] * 0.75 + next_point[0] * 0.25,
                    point[1] * 0.75 + next_point[1] * 0.25,
                )
            )
            smoothed.append(
                (
                    point[0] * 0.25 + next_point[0] * 0.75,
                    point[1] * 0.25 + next_point[1] * 0.75,
                )
            )
        points = smoothed
    return points


def _format_number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def region_contour_loops(
    region_labels: np.ndarray,
    region_id: int,
    bounds: tuple[int, int, int, int],
    smoothing_iterations: int = 0,
    min_smooth_area_px: float = 18.0,
) -> list[list[FloatPoint]]:
    """Retourne les boucles de contour, lissées si la zone est assez grande."""
    min_x, min_y, max_x, max_y = bounds
    local = (
        region_labels[min_y : max_y + 1, min_x : max_x + 1] == region_id
    )
    return [
        _smooth_loop(loop, smoothing_iterations, min_smooth_area_px)
        for loop in _trace_edges(_mask_edges(local, min_x, min_y))
    ]


def region_svg_path(
    region_labels: np.ndarray,
    region_id: int,
    bounds: tuple[int, int, int, int],
    smoothing_iterations: int = 0,
    min_smooth_area_px: float = 18.0,
) -> str:
    """Vectorise le contour d'une région, trous inclus."""
    loops = region_contour_loops(
        region_labels,
        region_id,
        bounds,
        smoothing_iterations=smoothing_iterations,
        min_smooth_area_px=min_smooth_area_px,
    )
    commands: list[str] = []
    for loop in loops:
        commands.append(
            f"M {_format_number(loop[0][0])} {_format_number(loop[0][1])}"
        )
        commands.extend(
            f"L {_format_number(x)} {_format_number(y)}" for x, y in loop[1:]
        )
        commands.append("Z")
    return " ".join(commands)


def _hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(value) for value in rgb))


def _legend_svg(result: PipelineResult, colored: bool) -> str:
    geometry = result.print_geometry
    item_width = 30.0
    columns = max(1, int(geometry.printable_width_mm // item_width))
    item_width = geometry.printable_width_mm / columns
    origin_x = geometry.margin_mm
    origin_y = geometry.legend_origin_y_mm + 4.0
    items = [
        (
            f'<text x="{origin_x:.3f}" y="{origin_y:.3f}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="3" '
            'font-weight="bold">Palette</text>'
        )
    ]
    for index, rgb in enumerate(result.palette_rgb):
        row, column = divmod(index, columns)
        x = origin_x + column * item_width
        y = origin_y + 3.0 + row * 7.0
        fill = _hex(rgb) if colored else "white"
        items.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="5" height="5" '
            f'fill="{fill}" stroke="black" stroke-width="0.2"/>'
        )
        items.append(
            f'<text x="{x + 7.0:.3f}" y="{y + 3.8:.3f}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="2.8">'
            f'{index + 1} · {_hex(rgb)}</text>'
        )
    return "\n".join(items)


def build_svg(result: PipelineResult, colored: bool) -> str:
    """Construit le modèle coloré ou la feuille de coloriage numérotée."""
    geometry = result.print_geometry
    render_profile = adaptive_render_profile(result)
    stroke_px = render_profile.line_width_mm / geometry.mm_per_pixel
    paths: list[str] = []
    for region in result.regions_after:
        path_data = region_svg_path(
            result.region_labels_after,
            region.region_id,
            (region.min_x, region.min_y, region.max_x, region.max_y),
            smoothing_iterations=render_profile.smoothing_iterations,
            min_smooth_area_px=render_profile.min_smooth_area_px,
        )
        if not path_data:
            continue
        fill = (
            _hex(result.palette_rgb[region.palette_index])
            if colored
            else "white"
        )
        paths.append(
            f'<path d="{path_data}" fill="{fill}" fill-rule="evenodd" '
            f'stroke="black" stroke-width="{stroke_px:.5f}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    labels: list[str] = []
    if not colored:
        for placement in result.label_placements:
            if placement.status != "placed":
                continue
            font_px = placement.font_size_mm / geometry.mm_per_pixel
            labels.append(
                f'<text x="{placement.x_px:.3f}" y="{placement.y_px:.3f}" '
                f'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="{font_px:.4f}" text-anchor="middle" '
                'dominant-baseline="central" fill="#222">'
                f"{placement.number}</text>"
            )

    subject_outline = ""
    if result.subject_mask is not None:
        mask_labels = result.subject_mask.astype(np.uint32)
        mask_path = region_svg_path(
            mask_labels,
            1,
            (0, 0, mask_labels.shape[1] - 1, mask_labels.shape[0] - 1),
            smoothing_iterations=render_profile.smoothing_iterations,
            min_smooth_area_px=render_profile.min_smooth_area_px,
        )
        if mask_path:
            subject_outline = (
                f'<path d="{mask_path}" fill="none" fill-rule="evenodd" '
                f'stroke="black" stroke-width="{stroke_px * 1.8:.5f}" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )

    title = html.escape(result.config.title, quote=True)
    scale = geometry.mm_per_pixel
    transform = (
        f"translate({geometry.image_origin_x_mm:.6f} "
        f"{geometry.image_origin_y_mm:.6f}) scale({scale:.8f})"
    )
    page_width = geometry.page_width_mm
    page_height = geometry.page_height_mm
    mode = "Modèle coloré" if colored else "Coloriage numéroté"
    title_text = (
        ""
        if result.config.palette_layout == "separate"
        else (
            f'<text x="{page_width / 2:.3f}" y="8" '
            'font-family="Arial, Helvetica, sans-serif" font-size="4" '
            f'font-weight="bold" text-anchor="middle">{title}</text>\n'
        )
    )
    legend = (
        ""
        if result.config.palette_layout == "separate"
        else _legend_svg(result, colored=True) + "\n"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width}mm" '
        f'height="{page_height}mm" viewBox="0 0 {page_width} {page_height}">\n'
        f'<title>{title} — {mode}</title>\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        + title_text
        + f'<g transform="{transform}">\n'
        + "\n".join(paths)
        + ("\n" + "\n".join(labels) if labels else "")
        + ("\n" + subject_outline if subject_outline else "")
        + "\n</g>\n"
        + legend
        + "\n</svg>\n"
    )


def build_palette_svg(result: PipelineResult) -> str:
    """Construit une page SVG dédiée à la palette."""
    geometry = result.print_geometry
    page_width = geometry.page_width_mm
    page_height = geometry.page_height_mm
    title = html.escape(result.config.title, quote=True)
    item_width = 42.0
    item_height = 12.0
    columns = max(1, int(geometry.printable_width_mm // item_width))
    item_width = geometry.printable_width_mm / columns
    origin_y = geometry.margin_mm + 22.0
    items: list[str] = []
    for index, rgb in enumerate(result.palette_rgb):
        row, column = divmod(index, columns)
        x = geometry.margin_mm + column * item_width
        y = origin_y + row * item_height
        items.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="7" height="7" '
            f'fill="{_hex(rgb)}" stroke="black" stroke-width="0.2"/>'
        )
        items.append(
            f'<text x="{x + 9.5:.3f}" y="{y + 4.6:.3f}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="3">'
            f'{index + 1} · {_hex(rgb)}</text>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width}mm" '
        f'height="{page_height}mm" viewBox="0 0 {page_width} {page_height}">\n'
        f'<title>{title} — Palette</title>\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{page_width / 2:.3f}" y="{geometry.margin_mm:.3f}" '
        'font-family="Arial, Helvetica, sans-serif" font-size="5" '
        'font-weight="bold" text-anchor="middle" dominant-baseline="hanging">'
        'Palette</text>\n'
        f'<text x="{page_width / 2:.3f}" y="{geometry.margin_mm + 7.0:.3f}" '
        'font-family="Arial, Helvetica, sans-serif" font-size="3" '
        f'text-anchor="middle" dominant-baseline="hanging">{title}</text>\n'
        + "\n".join(items)
        + "\n</svg>\n"
    )


def save_svgs(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    coloring_path = output_dir / "coloriage.svg"
    model_path = output_dir / "modele-couleur.svg"
    coloring_path.write_text(build_svg(result, colored=False), encoding="utf-8")
    model_path.write_text(build_svg(result, colored=True), encoding="utf-8")
    paths = {"coloring_svg": coloring_path, "model_svg": model_path}
    if result.config.palette_layout == "separate":
        palette_path = output_dir / "palette.svg"
        palette_path.write_text(build_palette_svg(result), encoding="utf-8")
        paths["palette_svg"] = palette_path
    return paths
