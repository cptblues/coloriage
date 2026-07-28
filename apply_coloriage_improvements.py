#!/usr/bin/env python3
"""Applique les améliorations de numérotation et de rendu au dépôt Nuance.

Usage, depuis la racine du dépôt :

    python apply_coloriage_improvements.py

Options :

    --check       vérifie uniquement que la migration est applicable
    --no-tests    applique la migration sans lancer les tests
    --repo PATH   cible un dépôt situé ailleurs

Le script ne crée aucun commit et ne pousse rien sur GitHub.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TARGET_FILES = (
    "app/page.tsx",
    "poc4/src/coloriage_lot3/export.py",
    "poc4/src/coloriage_lot3/labeling.py",
    "poc4/src/coloriage_lot3/svg.py",
    "poc4/tests/test_labeling.py",
    "poc4/tests/test_pipeline.py",
)


PATCH = r'''diff --git a/app/page.tsx b/app/page.tsx
index 8939d0c..59e50c0 100644
--- a/app/page.tsx
+++ b/app/page.tsx
@@ -45,6 +45,8 @@ type EngineResult = {
       labeling?: {
         placed_count?: number;
         skipped_count?: number;
+        coverage_percent?: number;
+        reduced_font_count?: number;
       };
     };
   };
@@ -1055,8 +1057,11 @@ export default function Home() {
                   zones
                 </div>
                 <div>
-                  <strong>{format}</strong>
-                  prêt à imprimer
+                  <strong>
+                    {engineResult?.stats?.result?.labeling?.placed_count ?? "—"} /{" "}
+                    {engineResult?.regionsAfter ?? "—"}
+                  </strong>
+                  zones numérotées
                 </div>
               </div>
               <button
diff --git a/poc4/src/coloriage_lot3/export.py b/poc4/src/coloriage_lot3/export.py
index 8ae10c6..59f803f 100644
--- a/poc4/src/coloriage_lot3/export.py
+++ b/poc4/src/coloriage_lot3/export.py
@@ -14,7 +14,7 @@ from scipy import ndimage
 
 from .pipeline import PipelineResult
 from .regions import AdjacencyEdge, Region, make_region_preview
-from .svg import region_contour_loops, save_svgs
+from .svg import adaptive_render_profile, region_contour_loops, save_svgs
 from .subject import mask_overlay
 
 
@@ -70,6 +70,16 @@ def build_stats(result: PipelineResult) -> dict[str, Any]:
         placement.status == "placed" for placement in result.label_placements
     )
     skipped = len(result.label_placements) - placed
+    placed_fonts = [
+        placement.font_size_mm
+        for placement in result.label_placements
+        if placement.status == "placed"
+    ]
+    reduced_font_count = sum(
+        font_size + 1e-9 < result.config.min_number_font_mm
+        for font_size in placed_fonts
+    )
+    render_profile = adaptive_render_profile(result)
     subject_stats: dict[str, Any] = dict(result.subject_metadata)
     detail_stats: dict[str, Any] = dict(result.detail_metadata)
     if result.subject_mask is not None:
@@ -111,7 +121,7 @@ def build_stats(result: PipelineResult) -> dict[str, Any]:
             }
         )
     return {
-        "schema_version": "3.3",
+        "schema_version": "3.4",
         "source": result.source_metadata,
         "subject": subject_stats,
         "detail": detail_stats,
@@ -151,6 +161,16 @@ def build_stats(result: PipelineResult) -> dict[str, Any]:
                     for placement in result.label_placements
                     if placement.status != "placed"
                 ],
+                "reduced_font_count": int(reduced_font_count),
+                "smallest_font_mm": (
+                    float(min(placed_fonts)) if placed_fonts else 0.0
+                ),
+            },
+            "rendering": {
+                "effective_line_width_mm": render_profile.line_width_mm,
+                "smoothing_iterations": render_profile.smoothing_iterations,
+                "min_smooth_area_px": render_profile.min_smooth_area_px,
+                "preview_supersampling": render_profile.preview_supersampling,
             },
         },
         "timings_ms": {
@@ -341,14 +361,15 @@ def _draw_print_contours(
     pixels_per_mm: float,
 ) -> None:
     geometry = result.print_geometry
-    line_width_px = max(1, round(result.config.line_width_mm * pixels_per_mm))
+    render_profile = adaptive_render_profile(result)
+    line_width_px = max(1, round(render_profile.line_width_mm * pixels_per_mm))
     for region in result.regions_after:
         loops = region_contour_loops(
             result.region_labels_after,
             region.region_id,
             (region.min_x, region.min_y, region.max_x, region.max_y),
-            smoothing_iterations=result.config.contour_smoothing_iterations,
-            min_smooth_area_px=result.config.min_contour_smooth_area_px,
+            smoothing_iterations=render_profile.smoothing_iterations,
+            min_smooth_area_px=render_profile.min_smooth_area_px,
         )
         for loop in loops:
             if len(loop) < 2:
@@ -384,6 +405,18 @@ def _make_print_preview(
 ) -> Image.Image:
     """Rend un aperçu bitmap de la page physique sans dépendance SVG externe."""
     geometry = result.print_geometry
+    output_pixels_per_mm = pixels_per_mm
+    output_page_size = (
+        round(geometry.page_width_mm * output_pixels_per_mm),
+        round(geometry.page_height_mm * output_pixels_per_mm),
+    )
+    render_profile = adaptive_render_profile(result)
+    supersampling = (
+        render_profile.preview_supersampling
+        if output_pixels_per_mm <= 6.0
+        else 1
+    )
+    pixels_per_mm = output_pixels_per_mm * supersampling
     page_size = (
         round(geometry.page_width_mm * pixels_per_mm),
         round(geometry.page_height_mm * pixels_per_mm),
@@ -471,6 +504,8 @@ def _make_print_preview(
                 font=legend_font,
                 anchor="lm",
             )
+    if supersampling > 1:
+        return canvas.resize(output_page_size, Image.Resampling.LANCZOS)
     return canvas
 
 
diff --git a/poc4/src/coloriage_lot3/labeling.py b/poc4/src/coloriage_lot3/labeling.py
index 391bcc4..d4d2285 100644
--- a/poc4/src/coloriage_lot3/labeling.py
+++ b/poc4/src/coloriage_lot3/labeling.py
@@ -60,9 +60,9 @@ def place_region_labels(
 ) -> list[LabelPlacement]:
     """Place chaque numéro au maximum de la transformée de distance.
 
-    Le numéro est réduit jusqu'à ``min_font_mm`` si nécessaire. Une région trop
-    étroite est explicitement marquée ``skipped`` au lieu de produire un numéro
-    illisible ou placé sur un contour.
+    ``min_font_mm`` est une taille de lisibilité recommandée, pas une limite
+    bloquante. Pour garantir une couverture de 100 %, le padding puis la police
+    sont réduits autant que nécessaire dans les régions très étroites.
     """
     if preferred_font_mm <= 0 or min_font_mm <= 0:
         raise ValueError("Les tailles de police doivent être positives")
@@ -96,11 +96,25 @@ def place_region_labels(
         local_y, local_x = candidates[candidate_index]
         x_px = float(xs.start + local_x + 0.5)
         y_px = float(ys.start + local_y + 0.5)
-        clearance_mm = max_distance_px * geometry.mm_per_pixel
+        # La transformée EDT mesure jusqu'au centre du premier pixel extérieur.
+        # Retirer un demi-pixel donne une estimation conservatrice de l'espace
+        # réellement disponible autour du centre du glyphe.
+        clearance_mm = max(
+            0.5 * geometry.mm_per_pixel,
+            (max_distance_px - 0.5) * geometry.mm_per_pixel,
+        )
         number = int(region_palette[region_id]) + 1
 
         digit_factor = max(1.0, 0.62 * len(str(number)))
-        usable_diameter_mm = max(0.0, 2.0 * clearance_mm - 2.0 * padding_mm)
+        available_diameter_mm = 2.0 * clearance_mm
+        adaptive_padding_mm = min(
+            padding_mm,
+            0.12 * available_diameter_mm,
+        )
+        usable_diameter_mm = max(
+            geometry.mm_per_pixel * 0.15,
+            available_diameter_mm - 2.0 * adaptive_padding_mm,
+        )
         target_font_mm = _area_scaled_font_mm(
             region,
             geometry,
@@ -111,28 +125,17 @@ def place_region_labels(
             target_font_mm,
             usable_diameter_mm / digit_factor,
         )
-        if fitted_font_mm + 1e-9 < min_font_mm:
-            placements.append(
-                LabelPlacement(
-                    region_id=region_id,
-                    palette_index=number - 1,
-                    number=number,
-                    status="skipped",
-                    reason="zone_trop_etroite",
-                    x_px=x_px,
-                    y_px=y_px,
-                    clearance_mm=clearance_mm,
-                    font_size_mm=0.0,
-                )
-            )
-            continue
         placements.append(
             LabelPlacement(
                 region_id=region_id,
                 palette_index=number - 1,
                 number=number,
                 status="placed",
-                reason="",
+                reason=(
+                    "police_reduite_sous_seuil"
+                    if fitted_font_mm + 1e-9 < min_font_mm
+                    else ""
+                ),
                 x_px=x_px,
                 y_px=y_px,
                 clearance_mm=clearance_mm,
diff --git a/poc4/src/coloriage_lot3/svg.py b/poc4/src/coloriage_lot3/svg.py
index e9efbd4..e9cbb5c 100644
--- a/poc4/src/coloriage_lot3/svg.py
+++ b/poc4/src/coloriage_lot3/svg.py
@@ -4,6 +4,7 @@ from __future__ import annotations
 
 import html
 import math
+from dataclasses import dataclass
 from pathlib import Path
 
 import numpy as np
@@ -15,6 +16,54 @@ FloatPoint = tuple[float, float]
 Edge = tuple[Point, Point]
 
 
+@dataclass(frozen=True)
+class RenderProfile:
+    """Paramètres de tracé dérivés de la taille imprimée et de la résolution."""
+
+    line_width_mm: float
+    smoothing_iterations: int
+    min_smooth_area_px: float
+    preview_supersampling: int
+
+
+def adaptive_render_profile(result: PipelineResult) -> RenderProfile:
+    """Adapte traits et lissage à la géométrie physique du document."""
+    geometry = result.print_geometry
+    page_scale = 1.12 if geometry.page_format == "a3" else 1.0
+    low_resolution_boost = max(
+        0.0,
+        min(0.06, (geometry.mm_per_pixel - 0.16) * 0.45),
+    )
+    line_width_mm = max(
+        0.18,
+        min(
+            0.42,
+            result.config.line_width_mm * page_scale + low_resolution_boost,
+        ),
+    )
+
+    smoothing_iterations = result.config.contour_smoothing_iterations
+    if geometry.mm_per_pixel >= 0.22:
+        smoothing_iterations = min(3, max(2, smoothing_iterations + 1))
+    else:
+        smoothing_iterations = max(1, smoothing_iterations)
+
+    min_physical_area_mm2 = max(
+        2.5,
+        min(8.0, result.config.min_region_area_mm2 * 0.4),
+    )
+    min_smooth_area_px = max(
+        result.config.min_contour_smooth_area_px,
+        min_physical_area_mm2 / max(geometry.pixel_area_mm2, 1e-9),
+    )
+    return RenderProfile(
+        line_width_mm=line_width_mm,
+        smoothing_iterations=smoothing_iterations,
+        min_smooth_area_px=min_smooth_area_px,
+        preview_supersampling=2,
+    )
+
+
 def _direction(start: Point, end: Point) -> int:
     dx = end[0] - start[0]
     dy = end[1] - start[1]
@@ -236,15 +285,16 @@ def _legend_svg(result: PipelineResult, colored: bool) -> str:
 def build_svg(result: PipelineResult, colored: bool) -> str:
     """Construit le modèle coloré ou la feuille de coloriage numérotée."""
     geometry = result.print_geometry
-    stroke_px = result.config.line_width_mm / geometry.mm_per_pixel
+    render_profile = adaptive_render_profile(result)
+    stroke_px = render_profile.line_width_mm / geometry.mm_per_pixel
     paths: list[str] = []
     for region in result.regions_after:
         path_data = region_svg_path(
             result.region_labels_after,
             region.region_id,
             (region.min_x, region.min_y, region.max_x, region.max_y),
-            smoothing_iterations=result.config.contour_smoothing_iterations,
-            min_smooth_area_px=result.config.min_contour_smooth_area_px,
+            smoothing_iterations=render_profile.smoothing_iterations,
+            min_smooth_area_px=render_profile.min_smooth_area_px,
         )
         if not path_data:
             continue
@@ -280,8 +330,8 @@ def build_svg(result: PipelineResult, colored: bool) -> str:
             mask_labels,
             1,
             (0, 0, mask_labels.shape[1] - 1, mask_labels.shape[0] - 1),
-            smoothing_iterations=result.config.contour_smoothing_iterations,
-            min_smooth_area_px=result.config.min_contour_smooth_area_px,
+            smoothing_iterations=render_profile.smoothing_iterations,
+            min_smooth_area_px=render_profile.min_smooth_area_px,
         )
         if mask_path:
             subject_outline = (
diff --git a/poc4/tests/test_labeling.py b/poc4/tests/test_labeling.py
index f4aa044..724cbc2 100644
--- a/poc4/tests/test_labeling.py
+++ b/poc4/tests/test_labeling.py
@@ -11,7 +11,7 @@ from coloriage_lot3.svg import region_svg_path
 
 
 class LabelingTests(unittest.TestCase):
-    def test_places_number_in_large_region_and_skips_thin_region(self) -> None:
+    def test_places_every_number_inside_including_thin_regions(self) -> None:
         labels = np.ones((20, 20), dtype=np.uint32)
         labels[:, 10:] = 2
         labels[:, 10] = 3
@@ -43,7 +43,11 @@ class LabelingTests(unittest.TestCase):
         statuses = {item.region_id: item.status for item in placements}
         self.assertEqual(statuses[1], "placed")
         self.assertEqual(statuses[2], "placed")
-        self.assertEqual(statuses[3], "skipped")
+        self.assertEqual(statuses[3], "placed")
+        thin = next(item for item in placements if item.region_id == 3)
+        self.assertGreater(thin.font_size_mm, 0.0)
+        self.assertLess(thin.font_size_mm, 1.8)
+        self.assertEqual(thin.reason, "police_reduite_sous_seuil")
 
     def test_number_size_grows_with_printed_area(self) -> None:
         labels = np.full((400, 400), 2, dtype=np.uint32)
@@ -78,7 +82,8 @@ class LabelingTests(unittest.TestCase):
         self.assertEqual(by_region[1].status, "placed")
         self.assertEqual(by_region[2].status, "placed")
         self.assertLess(by_region[1].font_size_mm, by_region[2].font_size_mm)
-        self.assertAlmostEqual(by_region[1].font_size_mm, 1.8)
+        self.assertGreater(by_region[1].font_size_mm, 0.0)
+        self.assertLessEqual(by_region[1].font_size_mm, 1.8)
         self.assertAlmostEqual(by_region[2].font_size_mm, 3.2)
 
     def test_svg_path_contains_closed_contours(self) -> None:
diff --git a/poc4/tests/test_pipeline.py b/poc4/tests/test_pipeline.py
index d23a15b..8f15cfa 100644
--- a/poc4/tests/test_pipeline.py
+++ b/poc4/tests/test_pipeline.py
@@ -44,7 +44,7 @@ class PipelineTests(unittest.TestCase):
                 self.assertTrue(path.is_file(), path)
 
             stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
-            self.assertEqual(stats["schema_version"], "3.3")
+            self.assertEqual(stats["schema_version"], "3.4")
             self.assertEqual(
                 stats["result"]["regions_after"]["below_area_threshold_count"],
                 0,
@@ -53,6 +53,15 @@ class PipelineTests(unittest.TestCase):
                 stats["result"]["labeling"]["placed_count"],
                 len(result.regions_after),
             )
+            self.assertEqual(stats["result"]["labeling"]["skipped_count"], 0)
+            self.assertEqual(
+                stats["result"]["labeling"]["coverage_percent"],
+                100.0,
+            )
+            self.assertIn(
+                "effective_line_width_mm",
+                stats["result"]["rendering"],
+            )
             pdf = paths["pdf_document"].read_bytes()
             self.assertTrue(pdf.startswith(b"%PDF"))
             self.assertGreaterEqual(pdf.count(b"/Type /Page"), 2)
'''


class MigrationError(RuntimeError):
    """Erreur contrôlée et compréhensible pendant la migration."""


def run(
    command: list[str],
    cwd: Path,
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and process.returncode != 0:
        raise MigrationError(
            f"La commande {' '.join(command)} a échoué :\n{process.stdout.strip()}"
        )
    return process


def resolve_repo(value: str) -> Path:
    repo = Path(value).expanduser().resolve()
    if not (repo / ".git").exists():
        raise MigrationError(f"{repo} n'est pas la racine d'un dépôt Git.")
    missing = [path for path in TARGET_FILES if not (repo / path).is_file()]
    if missing:
        raise MigrationError(
            "Le dépôt ne correspond pas à la structure Nuance attendue. "
            f"Fichiers absents : {', '.join(missing)}"
        )
    return repo


def patch_state(repo: Path) -> str:
    forward = run(
        ["git", "apply", "--check", "--whitespace=error-all", "-"],
        repo,
        input_text=PATCH,
        check=False,
    )
    if forward.returncode == 0:
        return "applicable"
    reverse = run(
        ["git", "apply", "--reverse", "--check", "-"],
        repo,
        input_text=PATCH,
        check=False,
    )
    if reverse.returncode == 0:
        return "already_applied"
    raise MigrationError(
        "La migration ne peut pas être appliquée automatiquement. "
        "Le dépôt a probablement évolué ou contient des modifications qui "
        "chevauchent les fichiers ciblés.\n\n"
        f"Détail Git :\n{forward.stdout.strip()}"
    )


def create_backup(repo: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = repo.parent / f"{repo.name}-backup-{timestamp}"
    for relative in TARGET_FILES:
        source = repo / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def run_tests(repo: Path) -> None:
    poc_root = repo / "poc4"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poc_root / "src")
    print("\nTests Python…")
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=poc_root,
        env=env,
        check=False,
    )
    if process.returncode != 0:
        raise MigrationError(
            "Les modifications sont appliquées, mais les tests Python ont échoué. "
            "Vous pouvez restaurer les fichiers depuis la sauvegarde indiquée."
        )

    npm = shutil.which("npm")
    if npm and (repo / "node_modules").is_dir():
        print("\nTests de l'interface…")
        process = subprocess.run([npm, "test"], cwd=repo, check=False)
        if process.returncode != 0:
            raise MigrationError(
                "Les tests Python passent, mais les tests de l'interface ont échoué."
            )
    else:
        print(
            "\nTests de l'interface non lancés : exécutez d'abord `npm install`, "
            "puis `npm test`."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Garantit un numéro intérieur par région et améliore automatiquement "
            "les contours du coloriage Nuance."
        )
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="racine du dépôt Git à modifier (défaut : dossier courant)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="vérifie la compatibilité sans modifier les fichiers",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="n'exécute pas les tests après application",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = resolve_repo(args.repo)
        state = patch_state(repo)
        if state == "already_applied":
            print("Les améliorations sont déjà présentes. Aucun fichier modifié.")
            return 0
        if args.check:
            print("Vérification réussie : la migration peut être appliquée.")
            return 0

        backup = create_backup(repo)
        print(f"Sauvegarde créée : {backup}")
        run(
            ["git", "apply", "--whitespace=error-all", "-"],
            repo,
            input_text=PATCH,
        )
        print("Migration appliquée aux 6 fichiers ciblés.")
        if not args.no_tests:
            run_tests(repo)
        print(
            "\nTerminé. Vérifiez le résultat avec `git diff`, puis créez votre "
            "branche et votre pull request quand vous êtes satisfait."
        )
        return 0
    except MigrationError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
