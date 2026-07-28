"""Post-segmentation cleanup focused on printable and labelable regions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .regions import MergeEvent, describe_regions, merge_small_regions


@dataclass(frozen=True)
class CleanMergeResult:
    region_labels: NDArray[np.uint32]
    region_palette: NDArray[np.int32]
    events: list[MergeEvent]
    forced_merges: int
    passes_executed: int


def _region_groups(
    region_labels: NDArray[np.uint32],
    subject_mask: NDArray[np.bool_] | None,
) -> NDArray[np.int8] | None:
    if subject_mask is None:
        return None
    count = int(region_labels.max()) + 1
    total = np.bincount(region_labels.ravel(), minlength=count)
    subject = np.bincount(
        region_labels[np.asarray(subject_mask, dtype=bool)].ravel(),
        minlength=count,
    )
    groups = np.zeros(count, dtype=np.int8)
    groups[1:] = (subject[1:] * 2 >= total[1:]).astype(np.int8)
    return groups


def improve_region_labelability(
    *,
    region_labels: NDArray[np.uint32],
    region_palette: NDArray[np.int32],
    palette_lab: NDArray[np.float64],
    mm_per_pixel: float,
    min_region_pixels: int,
    min_region_area_mm2: float,
    preferred_font_mm: float,
    min_font_mm: float,
    padding_mm: float,
    strategy: str,
    color_tolerance: float,
    subject_mask: NDArray[np.bool_] | None,
    passes: int = 2,
) -> CleanMergeResult:
    """Merge thin strips before falling back to microscopic label fonts."""
    current_labels = np.asarray(region_labels, dtype=np.uint32)
    current_palette = np.asarray(region_palette, dtype=np.int32)
    all_events: list[MergeEvent] = []
    forced_merges = 0
    passes_executed = 0

    for _ in range(max(0, passes)):
        palette_digits = max(1, len(str(max(1, len(palette_lab)))))
        global_required_mm = min_font_mm * max(1.0, 0.62 * palette_digits) + 2.0 * padding_mm
        regions = describe_regions(
            current_labels,
            current_palette,
            mm_per_pixel,
            min_region_pixels,
            global_required_mm,
        )
        count = int(current_labels.max()) + 1
        thresholds = np.full(count, min_region_pixels, dtype=np.int64)
        needs_merge = 0
        for region in regions:
            number = int(current_palette[region.region_id]) + 1
            digit_factor = max(1.0, 0.62 * len(str(number)))
            required_width = min_font_mm * digit_factor + 2.0 * padding_mm
            awkward = (
                region.compactness < 0.018
                and region.area_mm2 < max(4.0 * min_region_area_mm2, 28.0)
            )
            if region.max_thickness_mm + 1e-9 < required_width or awkward:
                thresholds[region.region_id] = max(
                    thresholds[region.region_id],
                    region.pixel_count + 1,
                )
                needs_merge += 1
        if needs_merge == 0:
            break

        result = merge_small_regions(
            region_labels=current_labels,
            region_palette=current_palette,
            palette_lab=palette_lab,
            min_region_pixels=min_region_pixels,
            strategy=strategy,
            color_tolerance=color_tolerance,
            region_min_pixels=thresholds,
            region_groups=_region_groups(current_labels, subject_mask),
        )
        if not result.events:
            break
        offset = len(all_events)
        all_events.extend(replace(event, step=offset + index + 1) for index, event in enumerate(result.events))
        forced_merges += result.forced_merges
        current_labels = result.region_labels
        current_palette = result.region_palette
        passes_executed += 1

    return CleanMergeResult(
        region_labels=current_labels,
        region_palette=current_palette,
        events=all_events,
        forced_merges=forced_merges,
        passes_executed=passes_executed,
    )
