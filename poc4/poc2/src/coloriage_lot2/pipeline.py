"""Pipeline principal du Lot 2."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageOps
from sklearn.cluster import KMeans

from .color import lab_to_rgb, rgb_to_lab
from .geometry import PrintGeometry, compute_print_geometry
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

    def validate(self) -> None:
        if not 2 <= self.colors <= 40:
            raise ValueError("colors doit être compris entre 2 et 40")
        if not 64 <= self.max_side <= 8000:
            raise ValueError("max_side doit être compris entre 64 et 8000")
        if self.sample_pixels < self.colors:
            raise ValueError("sample_pixels doit être supérieur au nombre de couleurs")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity doit valoir 4 ou 8")
        if self.segmentation not in ("components", "slic"):
            raise ValueError("segmentation doit valoir components ou slic")
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
    adjacency_before: list[AdjacencyEdge]
    adjacency_after: list[AdjacencyEdge]
    merge_events: list[MergeEvent]
    forced_merges: int
    recolored_pixels: int
    print_geometry: PrintGeometry
    timings_ms: dict[str, float]
    source_metadata: dict[str, Any]
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
    actual_colors = min(config.colors, len(unique_training))
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
    timings["normalization"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    lab_image = rgb_to_lab(normalized_rgb)
    pixels_lab = lab_image.reshape(-1, 3)
    palette_lab, flat_labels = _fit_palette(pixels_lab, config)
    palette_labels = flat_labels.reshape(normalized_rgb.shape[:2])
    palette_rgb = lab_to_rgb(palette_lab)
    quantized_rgb = palette_rgb[palette_labels]
    timings["quantization"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    segmented_palette_labels = segment_palette_labels(
        normalized_rgb=normalized_rgb,
        palette_labels=palette_labels,
        palette_size=len(palette_rgb),
        method=config.segmentation,
        superpixels=config.superpixels,
        compactness=config.compactness,
        smoothing_radius=config.smoothing_radius,
    )
    segmented_rgb = palette_rgb[segmented_palette_labels]
    region_labels_before, region_palette_before = extract_connected_regions(
        segmented_palette_labels,
        palette_size=len(palette_rgb),
        connectivity=config.connectivity,
    )
    timings["segmentation"] = (time.perf_counter() - start) * 1000.0

    geometry = compute_print_geometry(
        image_width_px=int(normalized_rgb.shape[1]),
        image_height_px=int(normalized_rgb.shape[0]),
        page_format=config.page_format,
        orientation=config.orientation,
        margin_mm=config.margin_mm,
        min_region_area_mm2=config.min_region_area_mm2,
    )

    start = time.perf_counter()
    adjacency_before = build_adjacency(region_labels_before)
    merge_result = merge_small_regions(
        region_labels=region_labels_before,
        region_palette=region_palette_before,
        palette_lab=palette_lab,
        min_region_pixels=geometry.min_region_pixels,
        strategy=config.merge_strategy,
        color_tolerance=config.color_tolerance,
    )
    region_labels_after = merge_result.region_labels
    region_palette_after = merge_result.region_palette
    merged_palette_labels = region_palette_after[region_labels_after]
    merged_rgb = palette_rgb[merged_palette_labels]
    adjacency_after = build_adjacency(region_labels_after)
    recolored_pixels = int(
        np.count_nonzero(merged_palette_labels != segmented_palette_labels)
    )
    timings["graph_and_merge"] = (time.perf_counter() - start) * 1000.0

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
    timings["metrics"] = (time.perf_counter() - start) * 1000.0
    timings["total"] = (time.perf_counter() - total_start) * 1000.0

    return PipelineResult(
        normalized_rgb=normalized_rgb,
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
        adjacency_before=adjacency_before,
        adjacency_after=adjacency_after,
        merge_events=merge_result.events,
        forced_merges=merge_result.forced_merges,
        recolored_pixels=recolored_pixels,
        print_geometry=geometry,
        timings_ms=timings,
        source_metadata=source_metadata,
        config=config,
    )
