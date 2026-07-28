"""Image-edge guidance for conservative region merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage import color, feature, filters, morphology, segmentation


@dataclass(frozen=True)
class BoundaryGuidance:
    boundary_pixels: int
    strength_sum: float
    peak_strength: float

    @property
    def mean_strength(self) -> float:
        return self.strength_sum / max(1, self.boundary_pixels)


def _normalize(values: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    scale = float(np.percentile(array, 96.0))
    if scale <= 1e-9:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def build_edge_strength_map(
    rgb: NDArray[np.uint8],
    *,
    subject_mask: NDArray[np.bool_] | None = None,
    detail_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Create a normalized semantic edge map in the range 0..1."""
    source = np.asarray(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(source)
    luminance = _normalize(filters.sobel(lab[..., 0]))
    chroma = _normalize(
        np.hypot(filters.sobel(lab[..., 1]), filters.sobel(lab[..., 2]))
    )
    gray = color.rgb2gray(source)
    canny = feature.canny(gray, sigma=1.35).astype(np.float32)
    strength = np.clip(0.67 * luminance + 0.23 * chroma + 0.10 * canny, 0.0, 1.0)
    strength = filters.gaussian(strength, sigma=0.65, preserve_range=True).astype(np.float32)

    protected_pixels = 0
    if subject_mask is not None:
        outline = segmentation.find_boundaries(
            np.asarray(subject_mask, dtype=bool),
            mode="thick",
        )
        outline = morphology.dilation(outline, morphology.disk(1))
        strength[outline] = np.maximum(strength[outline], 0.98)
        protected_pixels += int(np.count_nonzero(outline))

    if detail_mask is not None:
        details = np.asarray(detail_mask, dtype=bool)
        detail_outline = segmentation.find_boundaries(details, mode="thick")
        detail_outline = morphology.dilation(detail_outline, morphology.disk(1))
        strength[detail_outline] = np.maximum(strength[detail_outline], 0.90)
        strength[details] = np.maximum(strength[details], strength[details] * 1.12)
        protected_pixels += int(np.count_nonzero(detail_outline))

    return np.clip(strength, 0.0, 1.0).astype(np.float32), {
        "enabled": True,
        "mean_strength": float(np.mean(strength)),
        "p90_strength": float(np.percentile(strength, 90.0)),
        "strong_pixel_count": int(np.count_nonzero(strength >= 0.72)),
        "semantic_protected_pixels": int(protected_pixels),
        "subject_aware": subject_mask is not None,
        "detail_aware": detail_mask is not None,
    }


def measure_region_boundaries(
    region_labels: NDArray[np.integer],
    edge_strength_map: NDArray[np.floating],
) -> dict[tuple[int, int], BoundaryGuidance]:
    """Aggregate edge evidence for every 4-connected region boundary."""
    labels = np.asarray(region_labels)
    strength = np.asarray(edge_strength_map, dtype=np.float32)
    if labels.shape != strength.shape:
        raise ValueError("edge_strength_map doit avoir la taille de region_labels")
    region_count = int(labels.max())
    base = region_count + 1
    codes: list[NDArray[np.int64]] = []
    values: list[NDArray[np.float32]] = []

    left = labels[:, :-1]
    right = labels[:, 1:]
    mask = (left != right) & (left > 0) & (right > 0)
    if np.any(mask):
        a = np.minimum(left[mask], right[mask]).astype(np.int64)
        b = np.maximum(left[mask], right[mask]).astype(np.int64)
        codes.append(a * base + b)
        values.append(np.maximum(strength[:, :-1][mask], strength[:, 1:][mask]))

    top = labels[:-1, :]
    bottom = labels[1:, :]
    mask = (top != bottom) & (top > 0) & (bottom > 0)
    if np.any(mask):
        a = np.minimum(top[mask], bottom[mask]).astype(np.int64)
        b = np.maximum(top[mask], bottom[mask]).astype(np.int64)
        codes.append(a * base + b)
        values.append(np.maximum(strength[:-1, :][mask], strength[1:, :][mask]))

    if not codes:
        return {}
    joined_codes = np.concatenate(codes)
    joined_values = np.concatenate(values).astype(np.float64)
    unique, inverse, counts = np.unique(joined_codes, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=joined_values, minlength=len(unique))
    peaks = np.zeros(len(unique), dtype=np.float64)
    np.maximum.at(peaks, inverse, joined_values)
    output: dict[tuple[int, int], BoundaryGuidance] = {}
    for index, code in enumerate(unique):
        region_a = int(code // base)
        region_b = int(code % base)
        output[(region_a, region_b)] = BoundaryGuidance(
            boundary_pixels=int(counts[index]),
            strength_sum=float(sums[index]),
            peak_strength=float(peaks[index]),
        )
    return output
