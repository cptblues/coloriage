"""Conversion entre la grille de pixels et la taille physique imprimée."""

from __future__ import annotations

import math
from dataclasses import dataclass


PAGE_SIZES_MM = {
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
}


@dataclass(frozen=True)
class PrintGeometry:
    page_format: str
    orientation: str
    page_width_mm: float
    page_height_mm: float
    margin_mm: float
    printable_width_mm: float
    printable_height_mm: float
    reserved_bottom_mm: float
    image_width_mm: float
    image_height_mm: float
    image_origin_x_mm: float
    image_origin_y_mm: float
    legend_origin_y_mm: float
    mm_per_pixel: float
    pixel_area_mm2: float
    min_region_area_mm2: float
    min_region_pixels: int


def compute_print_geometry(
    image_width_px: int,
    image_height_px: int,
    page_format: str,
    orientation: str,
    margin_mm: float,
    min_region_area_mm2: float,
    reserved_bottom_mm: float = 0.0,
) -> PrintGeometry:
    """Calcule la surface physique d'un pixel pour une image ajustée à la page."""
    if page_format not in PAGE_SIZES_MM:
        raise ValueError("page_format doit valoir a4 ou a3")
    if orientation not in ("portrait", "landscape"):
        raise ValueError("orientation doit valoir portrait ou landscape")
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("Les dimensions de l'image doivent être positives")
    if margin_mm < 0:
        raise ValueError("margin_mm doit être positif ou nul")
    if min_region_area_mm2 <= 0:
        raise ValueError("min_region_area_mm2 doit être strictement positif")
    if reserved_bottom_mm < 0:
        raise ValueError("reserved_bottom_mm doit être positif ou nul")

    page_width, page_height = PAGE_SIZES_MM[page_format]
    if orientation == "landscape":
        page_width, page_height = page_height, page_width
    printable_width = page_width - 2.0 * margin_mm
    printable_height = page_height - 2.0 * margin_mm
    image_box_height = printable_height - reserved_bottom_mm
    if printable_width <= 0 or image_box_height <= 0:
        raise ValueError("Les marges ne laissent aucune surface imprimable")

    mm_per_pixel = min(
        printable_width / image_width_px,
        image_box_height / image_height_px,
    )
    pixel_area = mm_per_pixel**2
    min_pixels = max(1, math.ceil(min_region_area_mm2 / pixel_area))
    image_width_mm = image_width_px * mm_per_pixel
    image_height_mm = image_height_px * mm_per_pixel
    image_origin_x_mm = margin_mm + (printable_width - image_width_mm) / 2.0
    image_origin_y_mm = margin_mm + (image_box_height - image_height_mm) / 2.0
    return PrintGeometry(
        page_format=page_format,
        orientation=orientation,
        page_width_mm=page_width,
        page_height_mm=page_height,
        margin_mm=margin_mm,
        printable_width_mm=printable_width,
        printable_height_mm=printable_height,
        reserved_bottom_mm=reserved_bottom_mm,
        image_width_mm=image_width_mm,
        image_height_mm=image_height_mm,
        image_origin_x_mm=image_origin_x_mm,
        image_origin_y_mm=image_origin_y_mm,
        legend_origin_y_mm=margin_mm + image_box_height,
        mm_per_pixel=mm_per_pixel,
        pixel_area_mm2=pixel_area,
        min_region_area_mm2=min_region_area_mm2,
        min_region_pixels=min_pixels,
    )


def compute_legend_height_mm(
    page_format: str,
    orientation: str,
    margin_mm: float,
    palette_size: int,
) -> float:
    """Réserve une légende lisible sur une ou plusieurs lignes."""
    if palette_size <= 0:
        return 0.0
    if page_format not in PAGE_SIZES_MM:
        raise ValueError("page_format doit valoir a4 ou a3")
    page_width, page_height = PAGE_SIZES_MM[page_format]
    if orientation == "landscape":
        page_width, page_height = page_height, page_width
    printable_width = page_width - 2.0 * margin_mm
    columns = max(1, int(printable_width // 30.0))
    rows = math.ceil(palette_size / columns)
    return 9.0 + rows * 7.0
