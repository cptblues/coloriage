"""Création et normalisation d'un masque de sujet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


def _clean_mask(
    mask: NDArray[np.bool_],
    close_radius: int = 2,
) -> NDArray[np.bool_]:
    """Nettoie les petits trous sans déplacer fortement la silhouette."""
    if close_radius > 0:
        structure = ndimage.generate_binary_structure(2, 2)
        for _ in range(close_radius):
            mask = ndimage.binary_closing(mask, structure=structure)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        keep = np.flatnonzero(sizes >= max(16, int(mask.size * 0.0001)))
        keep = keep[keep != 0]
        if len(keep):
            mask = np.isin(labels, keep)
    return np.asarray(mask, dtype=bool)


def _validate_coverage(mask: NDArray[np.bool_]) -> float:
    coverage = float(100.0 * np.count_nonzero(mask) / mask.size)
    if coverage < 0.5:
        raise ValueError(
            "Le masque couvre moins de 0,5 % de l'image. "
            "Essayez un autre modèle ou fournissez --subject-mask."
        )
    if coverage > 98.5:
        raise ValueError(
            "Le masque couvre plus de 98,5 % de l'image. "
            "Le sujet n'a probablement pas été séparé du fond."
        )
    return coverage


def load_manual_mask(
    path: str | Path,
    target_size: tuple[int, int],
    threshold: int = 128,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Charge un masque blanc=sujet et le met à la taille de l'image."""
    mask_path = Path(path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Masque introuvable : {mask_path}")
    with Image.open(mask_path) as opened:
        source_size = opened.size
        if "A" in opened.getbands():
            alpha = opened.getchannel("A")
            grayscale = (
                alpha
                if alpha.getextrema()[0] < 255
                else ImageOps.grayscale(opened)
            )
        else:
            grayscale = ImageOps.grayscale(opened)
        resized = source_size != target_size
        if resized:
            grayscale = grayscale.resize(target_size, Image.Resampling.BILINEAR)
        mask = np.asarray(grayscale, dtype=np.uint8) >= threshold
    mask = _clean_mask(mask)
    coverage = _validate_coverage(mask)
    return mask, {
        "mode": "manual",
        "mask_path": str(mask_path),
        "mask_original_width": source_size[0],
        "mask_original_height": source_size[1],
        "mask_resized": resized,
        "coverage_percent": coverage,
    }


def generate_ai_mask(
    normalized_rgb: NDArray[np.uint8],
    model_name: str,
    threshold: int = 128,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Génère localement un masque via rembg ; import paresseux volontaire."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError(
            "Le mode IA nécessite les dépendances optionnelles. "
            "Installez-les avec : python -m pip install -e \".[ai]\""
        ) from exc

    source = Image.fromarray(normalized_rgb, "RGB")
    try:
        session = new_session(model_name)
        result = remove(
            source,
            session=session,
            only_mask=True,
            post_process_mask=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Échec du modèle local {model_name!r} : {exc}"
        ) from exc

    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result))
    grayscale = ImageOps.grayscale(result).filter(ImageFilter.MedianFilter(3))
    mask = np.asarray(grayscale, dtype=np.uint8) >= threshold
    mask = _clean_mask(mask)
    coverage = _validate_coverage(mask)
    return mask, {
        "mode": "ai",
        "model": model_name,
        "coverage_percent": coverage,
        "inference": "local",
    }


def mask_overlay(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Produit une vue de contrôle : sujet naturel, fond teinté en cyan."""
    output = rgb.astype(np.float32).copy()
    tint = np.array([50.0, 210.0, 225.0], dtype=np.float32)
    output[~mask] = output[~mask] * 0.48 + tint * 0.52
    boundary = mask ^ ndimage.binary_erosion(mask)
    output[boundary] = np.array([255.0, 40.0, 80.0])
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)
