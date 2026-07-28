"""Placement robuste des numéros à l'intérieur des régions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .geometry import PrintGeometry
from .regions import Region


@dataclass(frozen=True)
class LabelPlacement:
    region_id: int
    palette_index: int
    number: int
    status: str
    reason: str
    x_px: float
    y_px: float
    clearance_mm: float
    font_size_mm: float


def _area_scaled_font_mm(
    region: Region,
    geometry: PrintGeometry,
    preferred_font_mm: float,
    min_font_mm: float,
) -> float:
    """Calcule une taille de numéro progressive selon la surface imprimée."""
    reference_area_mm2 = max(
        geometry.min_region_area_mm2,
        geometry.pixel_area_mm2,
        1e-9,
    )
    area_ratio = max(0.0, region.area_mm2) / reference_area_mm2
    full_size_area_ratio = 12.0
    progress = (
        (math.sqrt(area_ratio) - 1.0) / (math.sqrt(full_size_area_ratio) - 1.0)
        if area_ratio > 1.0
        else 0.0
    )
    progress = min(1.0, max(0.0, progress))
    return min_font_mm + (preferred_font_mm - min_font_mm) * progress


def place_region_labels(
    region_labels: NDArray[np.integer],
    region_palette: NDArray[np.integer],
    regions: list[Region],
    geometry: PrintGeometry,
    preferred_font_mm: float,
    min_font_mm: float,
    padding_mm: float,
) -> list[LabelPlacement]:
    """Place chaque numéro au maximum de la transformée de distance.

    ``min_font_mm`` est une taille de lisibilité recommandée, pas une limite
    bloquante. Pour garantir une couverture de 100 %, le padding puis la police
    sont réduits autant que nécessaire dans les régions très étroites.
    """
    if preferred_font_mm <= 0 or min_font_mm <= 0:
        raise ValueError("Les tailles de police doivent être positives")
    if min_font_mm > preferred_font_mm:
        raise ValueError("min_font_mm ne peut pas dépasser preferred_font_mm")
    if padding_mm < 0:
        raise ValueError("padding_mm doit être positif ou nul")

    placements: list[LabelPlacement] = []
    objects = ndimage.find_objects(region_labels)
    region_by_id = {region.region_id: region for region in regions}

    for region_id, slices in enumerate(objects, start=1):
        if slices is None or region_id >= len(region_palette):
            continue
        ys, xs = slices
        local_mask = region_labels[ys, xs] == region_id
        distance = ndimage.distance_transform_edt(
            np.pad(local_mask, 1, mode="constant", constant_values=False)
        )[1:-1, 1:-1]
        max_distance_px = float(distance.max())
        candidates = np.argwhere(np.isclose(distance, max_distance_px))
        region = region_by_id[region_id]
        target = np.asarray(
            [region.centroid_y - ys.start, region.centroid_x - xs.start],
            dtype=np.float64,
        )
        candidate_index = int(
            np.argmin(np.sum((candidates.astype(np.float64) - target) ** 2, axis=1))
        )
        local_y, local_x = candidates[candidate_index]
        x_px = float(xs.start + local_x + 0.5)
        y_px = float(ys.start + local_y + 0.5)
        # La transformée EDT mesure jusqu'au centre du premier pixel extérieur.
        # Retirer un demi-pixel donne une estimation conservatrice de l'espace
        # réellement disponible autour du centre du glyphe.
        clearance_mm = max(
            0.5 * geometry.mm_per_pixel,
            (max_distance_px - 0.5) * geometry.mm_per_pixel,
        )
        number = int(region_palette[region_id]) + 1

        digit_factor = max(1.0, 0.62 * len(str(number)))
        available_diameter_mm = 2.0 * clearance_mm
        adaptive_padding_mm = min(
            padding_mm,
            0.12 * available_diameter_mm,
        )
        usable_diameter_mm = max(
            geometry.mm_per_pixel * 0.15,
            available_diameter_mm - 2.0 * adaptive_padding_mm,
        )
        target_font_mm = _area_scaled_font_mm(
            region,
            geometry,
            preferred_font_mm,
            min_font_mm,
        )
        fitted_font_mm = min(
            target_font_mm,
            usable_diameter_mm / digit_factor,
        )
        placements.append(
            LabelPlacement(
                region_id=region_id,
                palette_index=number - 1,
                number=number,
                status="placed",
                reason=(
                    "police_reduite_sous_seuil"
                    if fitted_font_mm + 1e-9 < min_font_mm
                    else ""
                ),
                x_px=x_px,
                y_px=y_px,
                clearance_mm=clearance_mm,
                font_size_mm=fitted_font_mm,
            )
        )
    return placements
