"""Placement robuste des numéros à l'intérieur des régions."""

from __future__ import annotations

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

    Le numéro est réduit jusqu'à ``min_font_mm`` si nécessaire. Une région trop
    étroite est explicitement marquée ``skipped`` au lieu de produire un numéro
    illisible ou placé sur un contour.
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
        clearance_mm = max_distance_px * geometry.mm_per_pixel
        number = int(region_palette[region_id]) + 1

        digit_factor = max(1.0, 0.62 * len(str(number)))
        usable_diameter_mm = max(0.0, 2.0 * clearance_mm - 2.0 * padding_mm)
        fitted_font_mm = min(
            preferred_font_mm,
            usable_diameter_mm / digit_factor,
        )
        if fitted_font_mm + 1e-9 < min_font_mm:
            placements.append(
                LabelPlacement(
                    region_id=region_id,
                    palette_index=number - 1,
                    number=number,
                    status="skipped",
                    reason="zone_trop_etroite",
                    x_px=x_px,
                    y_px=y_px,
                    clearance_mm=clearance_mm,
                    font_size_mm=0.0,
                )
            )
            continue
        placements.append(
            LabelPlacement(
                region_id=region_id,
                palette_index=number - 1,
                number=number,
                status="placed",
                reason="",
                x_px=x_px,
                y_px=y_px,
                clearance_mm=clearance_mm,
                font_size_mm=fitted_font_mm,
            )
        )
    return placements
