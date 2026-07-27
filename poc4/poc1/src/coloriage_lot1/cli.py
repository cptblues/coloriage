"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .export import build_stats, export_result
from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coloriage-lot1",
        description=(
            "Normalise une image, réduit sa palette en CIELAB avec K-means "
            "et extrait ses composantes connexes."
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
    parser.add_argument(
        "--max-side",
        type=int,
        default=1200,
        help="Taille maximale du plus grand côté (défaut : 1200)",
    )
    parser.add_argument(
        "--sample-pixels",
        type=int,
        default=100_000,
        help="Nombre maximal de pixels pour entraîner K-means (défaut : 100000)",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Voisinage des composantes connexes : 4 ou 8 (défaut : 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour un résultat reproductible (défaut : 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autoriser l'écriture dans un dossier de sortie non vide",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig(
        colors=args.colors,
        max_side=args.max_side,
        sample_pixels=args.sample_pixels,
        connectivity=args.connectivity,
        seed=args.seed,
    )
    try:
        result = run_pipeline(args.input, config)
        paths = export_result(result, args.output, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        parser.exit(2, f"Erreur : {exc}\n")

    stats = build_stats(result)
    summary = {
        "output": str(paths["overview"].parent),
        "colors": stats["result"]["actual_colors"],
        "regions": stats["result"]["region_count"],
        "smallest_region_pixels": stats["result"]["regions"]["min_pixels"],
        "duration_ms": stats["timings_ms"]["total"],
        "overview": str(paths["overview"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

