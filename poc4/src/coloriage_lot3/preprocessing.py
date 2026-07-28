"""Edge-preserving preparation used before palette fitting and segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage import color, filters, restoration


def _resize_mask(mask: NDArray[np.bool_] | None, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    if mask.shape != shape:
        raise ValueError("Le masque doit avoir la même taille que l'image normalisée")
    return np.asarray(mask, dtype=bool)


def prepare_processing_image(
    rgb: NDArray[np.uint8],
    *,
    subject_mask: NDArray[np.bool_] | None = None,
    detail_mask: NDArray[np.bool_] | None = None,
    sigma_color: float = 0.055,
    sigma_spatial: float = 3.0,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """Reduce photographic texture while preserving meaningful boundaries.

    The subject receives a moderate bilateral denoise. The background receives
    a stronger second pass. User-painted detail zones blend more of the source
    image back in so eyes, faces, fur, and small objects retain structure.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb doit être une image RGB")
    if not 0.001 <= sigma_color <= 0.25:
        raise ValueError("sigma_color doit être compris entre 0,001 et 0,25")
    if not 0.25 <= sigma_spatial <= 12.0:
        raise ValueError("sigma_spatial doit être compris entre 0,25 et 12")

    shape = rgb.shape[:2]
    subject_mask = _resize_mask(subject_mask, shape)
    detail_mask = _resize_mask(detail_mask, shape)
    source = rgb.astype(np.float32) / 255.0

    base = restoration.denoise_bilateral(
        source,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        bins=256,
        channel_axis=-1,
    )
    background = restoration.denoise_bilateral(
        base,
        sigma_color=min(0.25, sigma_color * 1.45),
        sigma_spatial=min(12.0, sigma_spatial * 1.65),
        bins=256,
        channel_axis=-1,
    )

    # Stabilize chroma more than lightness to avoid palette speckle while
    # retaining the geometry carried by luminance boundaries.
    lab = color.rgb2lab(np.clip(base, 0.0, 1.0))
    lab[..., 1] = filters.gaussian(lab[..., 1], sigma=0.65, preserve_range=True)
    lab[..., 2] = filters.gaussian(lab[..., 2], sigma=0.65, preserve_range=True)
    base = np.clip(color.lab2rgb(lab), 0.0, 1.0)

    output = background
    if subject_mask is not None:
        output = np.where(subject_mask[..., None], base, background)
    else:
        output = 0.70 * base + 0.30 * background

    if detail_mask is not None:
        detailed = 0.68 * source + 0.32 * base
        output = np.where(detail_mask[..., None], detailed, output)

    result = np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)
    return result, {
        "mode": "edge_preserving",
        "sigma_color": float(sigma_color),
        "sigma_spatial": float(sigma_spatial),
        "subject_aware": subject_mask is not None,
        "detail_aware": detail_mask is not None,
    }
