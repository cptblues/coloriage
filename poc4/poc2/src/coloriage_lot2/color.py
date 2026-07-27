"""Conversions colorimétriques sRGB ↔ CIELAB (illuminant D65)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
UInt8Array = NDArray[np.uint8]

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def _srgb_to_linear(rgb: FloatArray) -> FloatArray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(rgb: FloatArray) -> FloatArray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def rgb_to_lab(rgb_uint8: UInt8Array) -> FloatArray:
    """Convertit un tableau RGB uint8 (..., 3) vers CIELAB D65."""
    rgb = np.asarray(rgb_uint8, dtype=np.float64) / 255.0
    linear = _srgb_to_linear(rgb)
    xyz = linear @ _RGB_TO_XYZ.T
    ratio = xyz / _D65_WHITE
    threshold = _DELTA**3
    f = np.where(
        ratio > threshold,
        np.cbrt(ratio),
        ratio / (3.0 * _DELTA**2) + 4.0 / 29.0,
    )
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb(lab: FloatArray) -> UInt8Array:
    """Convertit un tableau CIELAB D65 (..., 3) vers RGB uint8."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz_ratio = np.where(
        f > _DELTA,
        f**3,
        3.0 * _DELTA**2 * (f - 4.0 / 29.0),
    )
    xyz = xyz_ratio * _D65_WHITE
    linear = xyz @ _XYZ_TO_RGB.T
    srgb = _linear_to_srgb(linear)
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)

