"""Interface en ligne de commande du Lot 2."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .export import build_stats, export_result, export_strategy_comparison
from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coloriage-lot2",
        description=(
            "Segmente une image, construit son graphe de régions et fusionne "
            "les zones trop petites pour le format imprimé."
        ),
    )
    parser.add_argument("input", type=Path, help="Image JPG ou PNG à analyser")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Dossier dans lequel écrire les résultats",
    )
    parser.add_argument(
        "-k",
        "--colors",
        type=int,
        default=12,
        help="Nombre de couleurs demandé, de 2 à 40 (défaut : 12)",
    )
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--sample-pixels", type=int, default=100_000)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--segmentation",
        choices=("components", "slic"),
        default="slic",
        help="Baseline par composantes ou segmentation SLIC (défaut : slic)",
    )
    parser.add_argument(
        "--superpixels",
        type=int,
        default=900,
        help="Nombre cible de superpixels SLIC (défaut : 900)",
    )
    parser.add_argument(
        "--compactness",
        type=float,
        default=10.0,
        help="Régularité spatiale de SLIC (défaut : 10)",
    )
    parser.add_argument(
        "--smoothing-radius",
        type=int,
        default=1,
        help="Rayon en pixels du lissage des contours, de 0 à 8 (défaut : 1)",
    )
    parser.add_argument(
        "--merge-strategy",
        choices=("color", "boundary", "balanced"),
        default="balanced",
        help="Règle de choix de la région voisine (défaut : balanced)",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Exécuter color, boundary et balanced dans trois sous-dossiers",
    )
    parser.add_argument(
        "--color-tolerance",
        type=float,
        default=35.0,
        help="Delta E au-delà duquel une fusion est signalée comme forcée",
    )
    parser.add_argument(
        "--page-format",
        choices=("a4", "a3"),
        default="a4",
        help="Format physique cible (défaut : a4)",
    )
    parser.add_argument(
        "--orientation",
        choices=("portrait", "landscape"),
        default="portrait",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=12.0,
        help="Marge physique autour de l'image (défaut : 12 mm)",
    )
    parser.add_argument(
        "--min-region-area-mm2",
        type=float,
        default=9.0,
        help="Surface imprimée minimale d'une zone (défaut : 9 mm²)",
    )
    parser.add_argument(
        "--thin-width-mm",
        type=float,
        default=1.5,
        help="Épaisseur sous laquelle une zone est signalée comme fine",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autoriser l'écriture dans un dossier de sortie non vide",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        colors=args.colors,
        max_side=args.max_side,
        sample_pixels=args.sample_pixels,
        connectivity=args.connectivity,
        seed=args.seed,
        segmentation=args.segmentation,
        superpixels=args.superpixels,
        compactness=args.compactness,
        smoothing_radius=args.smoothing_radius,
        merge_strategy=args.merge_strategy,
        color_tolerance=args.color_tolerance,
        page_format=args.page_format,
        orientation=args.orientation,
        margin_mm=args.margin_mm,
        min_region_area_mm2=args.min_region_area_mm2,
        thin_width_mm=args.thin_width_mm,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    try:
        if args.compare_strategies:
            destination = args.output.expanduser().resolve()
            if (
                destination.exists()
                and any(destination.iterdir())
                and not args.overwrite
            ):
                raise FileExistsError(
                    f"Le dossier de sortie n'est pas vide : {destination}. "
                    "Utilisez --overwrite pour le remplacer."
                )
            results = {}
            for strategy in ("color", "boundary", "balanced"):
                strategy_config = replace(config, merge_strategy=strategy)
                result = run_pipeline(args.input, strategy_config)
                export_result(
                    result,
                    destination / strategy,
                    overwrite=args.overwrite,
                )
                results[strategy] = result
            comparison = export_strategy_comparison(results, destination)
            summary = {
                "output": str(destination),
                "strategies": {
                    name: {
                        "regions_before": len(result.regions_before),
                        "regions_after": len(result.regions_after),
                        "merges": len(result.merge_events),
                        "duration_ms": round(result.timings_ms["total"], 3),
                    }
                    for name, result in results.items()
                },
                "comparison": str(comparison["image"]),
            }
        else:
            result = run_pipeline(args.input, config)
            paths = export_result(result, args.output, overwrite=args.overwrite)
            stats = build_stats(result)
            summary = {
                "output": str(paths["overview"].parent),
                "colors": stats["result"]["actual_colors"],
                "regions_before": len(result.regions_before),
                "regions_after": len(result.regions_after),
                "merges": len(result.merge_events),
                "min_region_pixels": result.print_geometry.min_region_pixels,
                "remaining_below_threshold": stats["result"]["regions_after"][
                    "below_area_threshold_count"
                ],
                "thin_regions": stats["result"]["regions_after"][
                    "thin_region_count"
                ],
                "duration_ms": stats["timings_ms"]["total"],
                "overview": str(paths["overview"]),
            }
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        parser.exit(2, f"Erreur : {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
