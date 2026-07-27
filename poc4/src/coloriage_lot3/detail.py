"""Masques des zones à traiter avec davantage de détail."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


def load_detail_mask(
    path: str | Path,
    target_size: tuple[int, int],
    threshold: int = 128,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Charge un masque blanc=zone détaillée et le met à la taille de l'image."""
    mask_path = Path(path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Masque de détail introuvable : {mask_path}")
    with Image.open(mask_path) as opened:
        source_size = opened.size
        if "A" in opened.getbands():
            grayscale = opened.getchannel("A")
        else:
            grayscale = ImageOps.grayscale(opened)
        resized = source_size != target_size
        if resized:
            grayscale = grayscale.resize(target_size, Image.Resampling.BILINEAR)
        mask = np.asarray(grayscale, dtype=np.uint8) >= threshold

    coverage = float(100.0 * np.count_nonzero(mask) / mask.size)
    if coverage <= 0:
        raise ValueError("Le masque de détail est vide.")
    if coverage > 95:
        raise ValueError("Le masque de détail couvre presque toute l'image.")

    return np.asarray(mask, dtype=bool), {
        "mode": "manual",
        "mask_path": str(mask_path),
        "mask_original_width": source_size[0],
        "mask_original_height": source_size[1],
        "mask_resized": resized,
        "coverage_percent": coverage,
    }
