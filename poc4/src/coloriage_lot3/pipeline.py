"""Pipeline principal du Lot 3."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageOps
from sklearn.cluster import KMeans

from .color import lab_to_rgb, rgb_to_lab
from .detail import load_detail_mask
from .edges import build_edge_strength_map
from .geometry import (
    PrintGeometry,
    compute_legend_height_mm,
    compute_print_geometry,
)
from .labeling import LabelPlacement, place_region_labels
from .lineart import build_line_art_mask
from .palette import build_global_palette, merge_near_palette_colors
from .preprocessing import prepare_processing_image
from .quality import improve_region_labelability
from .regions import (
    AdjacencyEdge,
    MergeEvent,
    Region,
    build_adjacency,
    describe_regions,
    extract_connected_regions,
    merge_small_regions,
)
from .segmentation import segment_palette_labels
from .subject import generate_ai_mask, load_manual_mask


@dataclass(frozen=True)
class PipelineConfig:
    colors: int = 12
    max_side: int = 1200
    sample_pixels: int = 100_000
    connectivity: int = 8
    seed: int = 42
    segmentation: str = "slic"
    superpixels: int = 900
    compactness: float = 10.0
    smoothing_radius: int = 1
    merge_strategy: str = "balanced"
    color_tolerance: float = 35.0
    page_format: str = "a4"
    orientation: str = "portrait"
    margin_mm: float = 12.0
    min_region_area_mm2: float = 9.0
    thin_width_mm: float = 1.5
    legend_height_mm: float = 0.0
    palette_layout: str = "inline"
    number_font_mm: float = 3.2
    min_number_font_mm: float = 1.8
    number_padding_mm: float = 0.45
    line_width_mm: float = 0.25
    auto_tune: bool = False
    contour_smoothing_iterations: int = 1
    min_contour_smooth_area_px: float = 18.0
    contour_simplify_mm: float = 0.10
    preprocess_sigma_color: float = 0.055
    preprocess_sigma_spatial: float = 3.0
    palette_merge_delta_e: float = 4.0
    palette_mode: str = "adaptive"
    edge_guided_merge: bool = True
    edge_merge_weight: float = 22.0
    edge_protection_threshold: float = 0.72
    thin_merge_passes: int = 2
    line_art_enabled: bool = True
    line_art_detail: float = 0.65
    subject_mode: str = "none"
    subject_mask_path: str | None = None
    ai_model: str = "birefnet-general"
    mask_threshold: int = 128
    subject_color_ratio: float = 0.68
    subject_min_region_area_mm2: float = 6.0
    background_min_region_area_mm2: float = 28.0
    background_superpixel_ratio: float = 0.45
    background_smoothing_radius: int = 3
    detail_mask_path: str | None = None
    detail_min_region_area_mm2: float = 4.0
    title: str = "Coloriage mystère"

    def validate(self) -> None:
        if not 2 <= self.colors <= 40:
            raise ValueError("colors doit être compris entre 2 et 40")
        if not 64 <= self.max_side <= 8000:
            raise ValueError("max_side doit être compris entre 64 et 8000")
        if self.sample_pixels < self.colors:
            raise ValueError("sample_pixels doit être supérieur au nombre de couleurs")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity doit valoir 4 ou 8")
        if self.segmentation not in ("components", "slic", "slic_legacy"):
            raise ValueError(
                "segmentation doit valoir components, slic ou slic_legacy"
            )
        if not 10 <= self.superpixels <= 20_000:
            raise ValueError("superpixels doit être compris entre 10 et 20000")
        if self.compactness <= 0:
            raise ValueError("compactness doit être strictement positif")
        if not 0 <= self.smoothing_radius <= 8:
            raise ValueError("smoothing_radius doit être compris entre 0 et 8")
        if self.merge_strategy not in ("color", "boundary", "balanced"):
            raise ValueError(
                "merge_strategy doit valoir color, boundary ou balanced"
            )
        if self.color_tolerance <= 0:
            raise ValueError("color_tolerance doit être strictement positif")
        if self.thin_width_mm <= 0:
            raise ValueError("thin_width_mm doit être strictement positif")
        if self.legend_height_mm < 0:
            raise ValueError("legend_height_mm doit être positif ou nul")
        if self.palette_layout not in ("inline", "separate"):
            raise ValueError("palette_layout doit valoir inline ou separate")
        if self.number_font_mm <= 0 or self.min_number_font_mm <= 0:
            raise ValueError("Les tailles de numéro doivent être positives")
        if self.min_number_font_mm > self.number_font_mm:
            raise ValueError(
                "min_number_font_mm ne peut pas dépasser number_font_mm"
            )
        if self.number_padding_mm < 0:
            raise ValueError("number_padding_mm doit être positif ou nul")
        if self.line_width_mm <= 0:
            raise ValueError("line_width_mm doit être strictement positif")
        if not 0 <= self.contour_smoothing_iterations <= 4:
            raise ValueError(
                "contour_smoothing_iterations doit être compris entre 0 et 4"
            )
        if self.min_contour_smooth_area_px < 0:
            raise ValueError("min_contour_smooth_area_px doit être positif ou nul")
        if not 0.0 <= self.contour_simplify_mm <= 1.0:
            raise ValueError("contour_simplify_mm doit être compris entre 0 et 1")
        if not 0.001 <= self.preprocess_sigma_color <= 0.25:
            raise ValueError("preprocess_sigma_color doit être compris entre 0,001 et 0,25")
        if not 0.25 <= self.preprocess_sigma_spatial <= 12.0:
            raise ValueError("preprocess_sigma_spatial doit être compris entre 0,25 et 12")
        if not 0.0 <= self.palette_merge_delta_e <= 20.0:
            raise ValueError("palette_merge_delta_e doit être compris entre 0 et 20")
        if self.palette_mode not in ("legacy", "adaptive", "exact"):
            raise ValueError("palette_mode doit valoir legacy, adaptive ou exact")
        if not 0.0 <= self.edge_merge_weight <= 100.0:
            raise ValueError("edge_merge_weight doit être compris entre 0 et 100")
        if not 0.0 <= self.edge_protection_threshold <= 1.0:
            raise ValueError("edge_protection_threshold doit être compris entre 0 et 1")
        if not 0 <= self.thin_merge_passes <= 5:
            raise ValueError("thin_merge_passes doit être compris entre 0 et 5")
        if not 0.0 <= self.line_art_detail <= 1.0:
            raise ValueError("line_art_detail doit être compris entre 0 et 1")
        if self.subject_mode not in ("none", "ai", "manual"):
            raise ValueError("subject_mode doit valoir none, ai ou manual")
        if self.subject_mode != "none" and self.colors < 4:
            raise ValueError("Le mode sujet nécessite au moins 4 couleurs")
        if self.subject_mode == "manual" and not self.subject_mask_path:
            raise ValueError("Le mode manual nécessite subject_mask_path")
        if not 1 <= self.mask_threshold <= 254:
            raise ValueError("mask_threshold doit être compris entre 1 et 254")
        if not 0.5 <= self.subject_color_ratio <= 0.85:
            raise ValueError("subject_color_ratio doit être compris entre 0,5 et 0,85")
        if self.subject_min_region_area_mm2 <= 0:
            raise ValueError("subject_min_region_area_mm2 doit être positif")
        if self.background_min_region_area_mm2 <= 0:
            raise ValueError("background_min_region_area_mm2 doit être positif")
        if not 0.1 <= self.background_superpixel_ratio <= 1.0:
            raise ValueError(
                "background_superpixel_ratio doit être compris entre 0,1 et 1"
            )
        if not 0 <= self.background_smoothing_radius <= 8:
            raise ValueError(
                "background_smoothing_radius doit être compris entre 0 et 8"
            )
        if self.detail_min_region_area_mm2 <= 0:
            raise ValueError("detail_min_region_area_mm2 doit être positif")


@dataclass(frozen=True)
class PipelineResult:
    normalized_rgb: NDArray[np.uint8]
    quantized_rgb: NDArray[np.uint8]
    segmented_rgb: NDArray[np.uint8]
    merged_rgb: NDArray[np.uint8]
    palette_labels: NDArray[np.int32]
    segmented_palette_labels: NDArray[np.int32]
    merged_palette_labels: NDArray[np.int32]
    region_labels_before: NDArray[np.uint32]
    region_labels_after: NDArray[np.uint32]
    region_palette_before: NDArray[np.int32]
    region_palette_after: NDArray[np.int32]
    palette_rgb: NDArray[np.uint8]
    palette_lab: NDArray[np.float64]
    regions_before: list[Region]
    regions_after: list[Region]
    label_placements: list[LabelPlacement]
    adjacency_before: list[AdjacencyEdge]
    adjacency_after: list[AdjacencyEdge]
    merge_events: list[MergeEvent]
    forced_merges: int
    recolored_pixels: int
    print_geometry: PrintGeometry
    timings_ms: dict[str, float]
    source_metadata: dict[str, Any]
    subject_mask: NDArray[np.bool_] | None
    subject_metadata: dict[str, Any]
    detail_mask: NDArray[np.bool_] | None
    detail_metadata: dict[str, Any]
    line_art_mask: NDArray[np.bool_] | None
    line_art_metadata: dict[str, Any]
    config: PipelineConfig


def _convert_to_srgb(image: Image.Image) -> Image.Image:
    """Convertit le profil ICC embarqué vers sRGB lorsque possible."""
    icc_bytes = image.info.get("icc_profile")
    if not icc_bytes:
        return image.convert("RGB")
    try:
        source_profile = ImageCms.ImageCmsProfile(
            __import__("io").BytesIO(icc_bytes)
        )
        target_profile = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(
            image.convert("RGB"),
            source_profile,
            target_profile,
            outputMode="RGB",
        )
    except (ImageCms.PyCMSError, OSError, ValueError):
        return image.convert("RGB")


def _load_and_normalize(
    input_path: Path,
    max_side: int,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    with Image.open(input_path) as opened:
        original_size = opened.size
        original_mode = opened.mode
        original_format = opened.format or input_path.suffix.lstrip(".").upper()
        transposed = ImageOps.exif_transpose(opened)

        if "A" in transposed.getbands():
            rgba = transposed.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            working = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            working = _convert_to_srgb(transposed)

        scale = min(1.0, max_side / max(working.size))
        if scale < 1.0:
            new_size = (
                max(1, round(working.width * scale)),
                max(1, round(working.height * scale)),
            )
            working = working.resize(new_size, Image.Resampling.LANCZOS)
        rgb = np.asarray(working, dtype=np.uint8).copy()

    metadata = {
        "input_path": str(input_path),
        "format": original_format,
        "original_mode": original_mode,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "normalized_width": int(rgb.shape[1]),
        "normalized_height": int(rgb.shape[0]),
        "resized": original_size != (int(rgb.shape[1]), int(rgb.shape[0])),
    }
    return rgb, metadata


def _fit_palette(
    pixels_lab: NDArray[np.float64],
    config: PipelineConfig,
    requested_colors: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    rng = np.random.default_rng(config.seed)
    pixel_count = pixels_lab.shape[0]
    sample_count = min(config.sample_pixels, pixel_count)
    if sample_count < pixel_count:
        sample_indices = rng.choice(pixel_count, size=sample_count, replace=False)
        training_pixels = pixels_lab[sample_indices]
    else:
        training_pixels = pixels_lab

    unique_training = np.unique(np.round(training_pixels, decimals=4), axis=0)
    actual_colors = min(requested_colors or config.colors, len(unique_training))
    if actual_colors < 2:
        centers = unique_training.astype(np.float64)
        labels = np.zeros(pixel_count, dtype=np.int32)
        return centers, labels

    model = KMeans(
        n_clusters=actual_colors,
        random_state=config.seed,
        n_init=10,
        algorithm="lloyd",
    )
    model.fit(training_pixels)
    labels = model.predict(pixels_lab).astype(np.int32)
    centers = model.cluster_centers_.astype(np.float64)
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return centers[order], inverse[labels].astype(np.int32)


def _region_subject_groups(
    region_labels: NDArray[np.uint32],
    subject_mask: NDArray[np.bool_],
) -> NDArray[np.int8]:
    """Classe chaque région par majorité de pixels dans le masque sujet."""
    count = int(region_labels.max()) + 1
    total = np.bincount(region_labels.ravel(), minlength=count)
    subject = np.bincount(
        region_labels[subject_mask].ravel(),
        minlength=count,
    )
    groups = np.zeros(count, dtype=np.int8)
    groups[1:] = (subject[1:] * 2 >= total[1:]).astype(np.int8)
    return groups


def _region_mask_overlaps(
    region_labels: NDArray[np.uint32],
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Repère les régions touchées par un masque, même partiellement."""
    count = int(region_labels.max()) + 1
    overlaps = np.zeros(count, dtype=bool)
    touched = np.unique(region_labels[mask])
    overlaps[touched] = True
    overlaps[0] = False
    return overlaps


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _image_adaptive_config(
    config: PipelineConfig,
    width: int,
    height: int,
) -> tuple[PipelineConfig, dict[str, Any]]:
    long_side = max(width, height)
    pixel_count = width * height
    if long_side >= 1400 or pixel_count >= 1_400_000:
        size_class = "large"
        superpixel_scale = 1.22
        min_area_scale = 0.90
        subject_area_scale = 0.90
        background_area_scale = 0.95
        detail_area_scale = 0.75
        smoothing_radius = max(config.smoothing_radius, 1)
        background_smoothing_radius = max(config.background_smoothing_radius, 2)
        contour_smoothing_iterations = max(config.contour_smoothing_iterations, 1)
        min_contour_smooth_area_px = max(config.min_contour_smooth_area_px, 18.0)
    elif long_side <= 900 or pixel_count <= 600_000:
        size_class = "small"
        superpixel_scale = 0.72
        min_area_scale = 1.35
        subject_area_scale = 1.15
        background_area_scale = 1.20
        detail_area_scale = 1.0
        smoothing_radius = max(config.smoothing_radius, 2)
        background_smoothing_radius = max(config.background_smoothing_radius, 4)
        contour_smoothing_iterations = max(config.contour_smoothing_iterations, 2)
        min_contour_smooth_area_px = max(config.min_contour_smooth_area_px, 24.0)
    else:
        size_class = "medium"
        superpixel_scale = 1.0
        min_area_scale = 1.0
        subject_area_scale = 1.0
        background_area_scale = 1.0
        detail_area_scale = 1.0
        smoothing_radius = max(config.smoothing_radius, 1)
        background_smoothing_radius = config.background_smoothing_radius
        contour_smoothing_iterations = max(config.contour_smoothing_iterations, 1)
        min_contour_smooth_area_px = config.min_contour_smooth_area_px

    if not config.auto_tune:
        return config, {
            "enabled": False,
            "size_class": size_class,
            "width": width,
            "height": height,
        }

    tuned = replace(
        config,
        superpixels=max(
            10,
            min(20_000, round(config.superpixels * superpixel_scale)),
        ),
        min_region_area_mm2=_clamp(
            config.min_region_area_mm2 * min_area_scale,
            1.0,
            80.0,
        ),
        subject_min_region_area_mm2=_clamp(
            config.subject_min_region_area_mm2 * subject_area_scale,
            1.0,
            80.0,
        ),
        background_min_region_area_mm2=_clamp(
            config.background_min_region_area_mm2 * background_area_scale,
            1.0,
            120.0,
        ),
        detail_min_region_area_mm2=_clamp(
            config.detail_min_region_area_mm2 * detail_area_scale,
            0.75,
            40.0,
        ),
        smoothing_radius=smoothing_radius,
        background_smoothing_radius=background_smoothing_radius,
        contour_smoothing_iterations=contour_smoothing_iterations,
        min_contour_smooth_area_px=min_contour_smooth_area_px,
    )
    return tuned, {
        "enabled": True,
        "size_class": size_class,
        "width": width,
        "height": height,
        "superpixel_scale": superpixel_scale,
        "min_area_scale": min_area_scale,
        "subject_area_scale": subject_area_scale,
        "background_area_scale": background_area_scale,
        "detail_area_scale": detail_area_scale,
    }


def run_pipeline(input_path: str | Path, config: PipelineConfig) -> PipelineResult:
    """Exécute segmentation, graphe et fusion en mémoire."""
    config.validate()
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image introuvable : {path}")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    normalized_rgb, source_metadata = _load_and_normalize(path, config.max_side)
    config, adaptive_profile = _image_adaptive_config(
        config,
        int(normalized_rgb.shape[1]),
        int(normalized_rgb.shape[0]),
    )
    config.validate()
    source_metadata["adaptive_profile"] = adaptive_profile
    timings["normalization"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    subject_mask: NDArray[np.bool_] | None = None
    subject_metadata: dict[str, Any] = {"mode": "none"}
    if config.subject_mode == "manual":
        subject_mask, subject_metadata = load_manual_mask(
            config.subject_mask_path or "",
            (normalized_rgb.shape[1], normalized_rgb.shape[0]),
            threshold=config.mask_threshold,
        )
    elif config.subject_mode == "ai":
        subject_mask, subject_metadata = generate_ai_mask(
            normalized_rgb,
            model_name=config.ai_model,
            threshold=config.mask_threshold,
        )
    timings["subject_mask"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    detail_mask: NDArray[np.bool_] | None = None
    detail_metadata: dict[str, Any] = {"mode": "none"}
    if config.detail_mask_path:
        detail_mask, detail_metadata = load_detail_mask(
            config.detail_mask_path,
            (normalized_rgb.shape[1], normalized_rgb.shape[0]),
        )
    timings["detail_mask"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    processing_rgb, processing_metadata = prepare_processing_image(
        normalized_rgb,
        subject_mask=subject_mask,
        detail_mask=detail_mask,
        sigma_color=config.preprocess_sigma_color,
        sigma_spatial=config.preprocess_sigma_spatial,
    )
    source_metadata["processing"] = processing_metadata
    lab_image = rgb_to_lab(processing_rgb)
    pixels_lab = lab_image.reshape(-1, 3)
    palette_cleanup: dict[str, Any]
    subject_palette_size = 0
    if subject_mask is None:
        initial_centers, initial_labels = _fit_palette(pixels_lab, config)
        if config.palette_mode == "legacy":
            palette_lab, flat_labels, palette_cleanup = merge_near_palette_colors(
                initial_centers,
                initial_labels,
                threshold=config.palette_merge_delta_e,
                minimum_colors=2,
            )
            palette_cleanup = {"mode": "legacy", **palette_cleanup}
        else:
            global_palette = build_global_palette(
                pixels_lab,
                initial_centers,
                initial_labels,
                requested_colors=config.colors,
                mode=config.palette_mode,
                merge_threshold=config.palette_merge_delta_e,
                importance_weights=(
                    np.where(detail_mask.ravel(), 1.5, 1.0)
                    if detail_mask is not None
                    else None
                ),
            )
            palette_lab = global_palette.centers_lab
            flat_labels = global_palette.labels
            palette_cleanup = global_palette.metadata
    else:
        flat_mask = subject_mask.ravel()
        subject_requested = max(2, round(config.colors * config.subject_color_ratio))
        background_requested = max(2, config.colors - subject_requested)
        if subject_requested + background_requested > config.colors:
            subject_requested = max(2, config.colors - background_requested)
        subject_lab, subject_labels = _fit_palette(
            pixels_lab[flat_mask],
            config,
            requested_colors=subject_requested,
        )
        background_lab, background_labels = _fit_palette(
            pixels_lab[~flat_mask],
            config,
            requested_colors=background_requested,
        )
        if config.palette_mode == "legacy":
            subject_lab, subject_labels, subject_cleanup = merge_near_palette_colors(
                subject_lab,
                subject_labels,
                threshold=config.palette_merge_delta_e,
                minimum_colors=2,
            )
            background_lab, background_labels, background_cleanup = merge_near_palette_colors(
                background_lab,
                background_labels,
                threshold=config.palette_merge_delta_e,
                minimum_colors=2,
            )
            subject_palette_size = len(subject_lab)
            palette_lab = np.concatenate([subject_lab, background_lab], axis=0)
            flat_labels = np.empty(len(pixels_lab), dtype=np.int32)
            flat_labels[flat_mask] = subject_labels
            flat_labels[~flat_mask] = background_labels + subject_palette_size
            palette_cleanup = {
                "mode": "legacy",
                "subject": subject_cleanup,
                "background": background_cleanup,
                "before": int(subject_cleanup["before"] + background_cleanup["before"]),
                "after": int(subject_cleanup["after"] + background_cleanup["after"]),
                "merged": int(subject_cleanup["merged"] + background_cleanup["merged"]),
                "threshold": float(config.palette_merge_delta_e),
            }
        else:
            initial_centers = np.concatenate([subject_lab, background_lab], axis=0)
            initial_labels = np.empty(len(pixels_lab), dtype=np.int32)
            initial_labels[flat_mask] = subject_labels
            initial_labels[~flat_mask] = background_labels + len(subject_lab)
            importance = np.where(flat_mask, 1.15, 1.0).astype(np.float64)
            if detail_mask is not None:
                importance[detail_mask.ravel()] *= 1.45
            global_palette = build_global_palette(
                pixels_lab,
                initial_centers,
                initial_labels,
                requested_colors=config.colors,
                mode=config.palette_mode,
                merge_threshold=config.palette_merge_delta_e,
                importance_weights=importance,
            )
            palette_lab = global_palette.centers_lab
            flat_labels = global_palette.labels
            palette_cleanup = global_palette.metadata
        subject_metadata.update(
            {
                "subject_colors": int(len(np.unique(flat_labels[flat_mask]))),
                "background_colors": int(len(np.unique(flat_labels[~flat_mask]))),
                "shared_colors": int(
                    len(
                        np.intersect1d(
                            np.unique(flat_labels[flat_mask]),
                            np.unique(flat_labels[~flat_mask]),
                        )
                    )
                ),
            }
        )
    source_metadata["palette_cleanup"] = palette_cleanup
    palette_labels = flat_labels.reshape(normalized_rgb.shape[:2])
    palette_rgb = lab_to_rgb(palette_lab)
    quantized_rgb = palette_rgb[palette_labels]
    timings["quantization"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    if subject_mask is None and detail_mask is None:
        segmented_palette_labels = segment_palette_labels(
            normalized_rgb=processing_rgb,
            palette_labels=palette_labels,
            palette_size=len(palette_rgb),
            method=config.segmentation,
            superpixels=config.superpixels,
            compactness=config.compactness,
            smoothing_radius=config.smoothing_radius,
        )
    else:
        fine_segmented = segment_palette_labels(
            normalized_rgb=processing_rgb,
            palette_labels=palette_labels,
            palette_size=len(palette_rgb),
            method=config.segmentation,
            superpixels=config.superpixels,
            compactness=config.compactness,
            smoothing_radius=config.smoothing_radius,
        )
        coarse_segmented = segment_palette_labels(
            normalized_rgb=processing_rgb,
            palette_labels=palette_labels,
            palette_size=len(palette_rgb),
            method=config.segmentation,
            superpixels=max(
                10,
                round(config.superpixels * config.background_superpixel_ratio),
            ),
            compactness=config.compactness,
            smoothing_radius=config.background_smoothing_radius,
        )
        if subject_mask is not None:
            if config.palette_mode == "legacy":
                subject_segmented = np.where(
                    fine_segmented < subject_palette_size,
                    fine_segmented,
                    palette_labels,
                )
                background_segmented = np.where(
                    coarse_segmented >= subject_palette_size,
                    coarse_segmented,
                    palette_labels,
                )
                segmented_palette_labels = np.where(
                    subject_mask,
                    subject_segmented,
                    background_segmented,
                ).astype(np.int32)
            else:
                segmented_palette_labels = np.where(
                    subject_mask,
                    fine_segmented,
                    coarse_segmented,
                ).astype(np.int32)
        else:
            segmented_palette_labels = coarse_segmented.astype(np.int32)
        if detail_mask is not None:
            segmented_palette_labels = np.where(
                detail_mask,
                fine_segmented,
                segmented_palette_labels,
            ).astype(np.int32)
    segmented_rgb = palette_rgb[segmented_palette_labels]
    region_labels_before, region_palette_before = extract_connected_regions(
        segmented_palette_labels,
        palette_size=len(palette_rgb),
        connectivity=config.connectivity,
    )
    timings["segmentation"] = (time.perf_counter() - start) * 1000.0

    edge_start = time.perf_counter()
    edge_strength_map, edge_metadata = build_edge_strength_map(
        processing_rgb,
        subject_mask=subject_mask,
        detail_mask=detail_mask,
    )
    source_metadata["edge_guidance"] = {
        **edge_metadata,
        "merge_enabled": bool(config.edge_guided_merge),
        "weight": float(config.edge_merge_weight),
        "protection_threshold": float(config.edge_protection_threshold),
    }
    timings["edge_guidance"] = (time.perf_counter() - edge_start) * 1000.0

    legend_height_mm = (
        0.0
        if config.palette_layout == "separate"
        else (
            config.legend_height_mm
            or compute_legend_height_mm(
                page_format=config.page_format,
                orientation=config.orientation,
                margin_mm=config.margin_mm,
                palette_size=len(palette_rgb),
            )
        )
    )
    geometry = compute_print_geometry(
        image_width_px=int(normalized_rgb.shape[1]),
        image_height_px=int(normalized_rgb.shape[0]),
        page_format=config.page_format,
        orientation=config.orientation,
        margin_mm=config.margin_mm,
        min_region_area_mm2=config.min_region_area_mm2,
        reserved_bottom_mm=legend_height_mm,
    )

    start = time.perf_counter()
    adjacency_before = build_adjacency(region_labels_before)
    region_thresholds = None
    region_groups = None
    if subject_mask is not None:
        region_groups = _region_subject_groups(region_labels_before, subject_mask)
        subject_min_pixels = max(
            1,
            int(
                np.ceil(
                    config.subject_min_region_area_mm2 / geometry.mm_per_pixel**2
                )
            ),
        )
        background_min_pixels = max(
            1,
            int(
                np.ceil(
                    config.background_min_region_area_mm2
                    / geometry.mm_per_pixel**2
                )
            ),
        )
        region_thresholds = np.where(
            region_groups == 1,
            subject_min_pixels,
            background_min_pixels,
        ).astype(np.int64)
    if detail_mask is not None:
        detail_min_pixels = max(
            1,
            int(
                np.ceil(
                    config.detail_min_region_area_mm2 / geometry.mm_per_pixel**2
                )
            ),
        )
        detail_overlaps = _region_mask_overlaps(region_labels_before, detail_mask)
        if region_thresholds is None:
            region_thresholds = np.full(
                int(region_labels_before.max()) + 1,
                geometry.min_region_pixels,
                dtype=np.int64,
            )
        region_thresholds = np.where(
            detail_overlaps,
            np.minimum(region_thresholds, detail_min_pixels),
            region_thresholds,
        ).astype(np.int64)
    merge_result = merge_small_regions(
        region_labels=region_labels_before,
        region_palette=region_palette_before,
        palette_lab=palette_lab,
        min_region_pixels=geometry.min_region_pixels,
        strategy=config.merge_strategy,
        color_tolerance=config.color_tolerance,
        region_min_pixels=region_thresholds,
        region_groups=region_groups,
        edge_strength_map=(edge_strength_map if config.edge_guided_merge else None),
        edge_weight=(config.edge_merge_weight if config.edge_guided_merge else 0.0),
        edge_protection_threshold=config.edge_protection_threshold,
    )
    region_labels_after = merge_result.region_labels
    region_palette_after = merge_result.region_palette
    clean_merge = improve_region_labelability(
        region_labels=region_labels_after,
        region_palette=region_palette_after,
        palette_lab=palette_lab,
        mm_per_pixel=geometry.mm_per_pixel,
        min_region_pixels=geometry.min_region_pixels,
        min_region_area_mm2=config.min_region_area_mm2,
        preferred_font_mm=config.number_font_mm,
        min_font_mm=config.min_number_font_mm,
        padding_mm=config.number_padding_mm,
        strategy=config.merge_strategy,
        color_tolerance=config.color_tolerance,
        subject_mask=subject_mask,
        edge_strength_map=(edge_strength_map if config.edge_guided_merge else None),
        edge_weight=(config.edge_merge_weight if config.edge_guided_merge else 0.0),
        edge_protection_threshold=config.edge_protection_threshold,
        passes=config.thin_merge_passes,
    )
    region_labels_after = clean_merge.region_labels
    region_palette_after = clean_merge.region_palette
    merge_events = [
        *merge_result.events,
        *(
            replace(event, step=len(merge_result.events) + index + 1)
            for index, event in enumerate(clean_merge.events)
        ),
    ]
    forced_merges = merge_result.forced_merges + clean_merge.forced_merges
    source_metadata["clean_merge"] = {
        "passes_executed": clean_merge.passes_executed,
        "extra_merges": len(clean_merge.events),
        "high_edge_merges": int(
            sum(event.edge_protected for event in merge_events)
        ),
        "mean_selected_edge_strength": float(
            np.mean([event.mean_edge_strength for event in merge_events])
            if merge_events
            else 0.0
        ),
    }
    merged_palette_labels = region_palette_after[region_labels_after]
    merged_rgb = palette_rgb[merged_palette_labels]
    adjacency_after = build_adjacency(region_labels_after)
    recolored_pixels = int(
        np.count_nonzero(merged_palette_labels != segmented_palette_labels)
    )
    timings["graph_and_merge"] = (time.perf_counter() - start) * 1000.0
    line_art_mask: NDArray[np.bool_] | None = None
    line_art_metadata: dict[str, Any] = {"enabled": False}
    if config.line_art_enabled:
        line_start = time.perf_counter()
        line_art_mask, line_art_metadata = build_line_art_mask(
            processing_rgb,
            subject_mask=subject_mask,
            detail_mask=detail_mask,
            region_labels=region_labels_after,
            detail_strength=config.line_art_detail,
        )
        timings["line_art"] = (time.perf_counter() - line_start) * 1000.0

    start = time.perf_counter()
    regions_before = describe_regions(
        region_labels_before,
        region_palette_before,
        geometry.mm_per_pixel,
        geometry.min_region_pixels,
        config.thin_width_mm,
    )
    regions_after = describe_regions(
        region_labels_after,
        region_palette_after,
        geometry.mm_per_pixel,
        geometry.min_region_pixels,
        config.thin_width_mm,
    )
    label_placements = place_region_labels(
        region_labels_after,
        region_palette_after,
        regions_after,
        geometry,
        preferred_font_mm=config.number_font_mm,
        min_font_mm=config.min_number_font_mm,
        padding_mm=config.number_padding_mm,
    )
    timings["metrics"] = (time.perf_counter() - start) * 1000.0
    timings["total"] = (time.perf_counter() - total_start) * 1000.0

    return PipelineResult(
        normalized_rgb=processing_rgb,
        quantized_rgb=quantized_rgb,
        segmented_rgb=segmented_rgb,
        merged_rgb=merged_rgb,
        palette_labels=palette_labels,
        segmented_palette_labels=segmented_palette_labels,
        merged_palette_labels=merged_palette_labels,
        region_labels_before=region_labels_before,
        region_labels_after=region_labels_after,
        region_palette_before=region_palette_before,
        region_palette_after=region_palette_after,
        palette_rgb=palette_rgb,
        palette_lab=palette_lab,
        regions_before=regions_before,
        regions_after=regions_after,
        label_placements=label_placements,
        adjacency_before=adjacency_before,
        adjacency_after=adjacency_after,
        merge_events=merge_events,
        forced_merges=forced_merges,
        recolored_pixels=recolored_pixels,
        print_geometry=geometry,
        timings_ms=timings,
        source_metadata=source_metadata,
        subject_mask=subject_mask,
        subject_metadata=subject_metadata,
        detail_mask=detail_mask,
        detail_metadata=detail_metadata,
        line_art_mask=line_art_mask,
        line_art_metadata=line_art_metadata,
        config=config,
    )
