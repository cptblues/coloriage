# POC Coloriage mystère — Lot 2

Ce projet transforme la baseline du Lot 1 en **régions coloriables mesurées à
l’échelle du papier**.

Le pipeline :

1. normalise une image JPG ou PNG ;
2. réduit sa palette en CIELAB avec K-means ;
3. segmente l’image avec une implémentation SLIC incluse dans le projet, ou
   conserve la baseline par composantes ;
4. lisse les contours ;
5. construit le graphe de voisinage des régions ;
6. convertit un seuil physique en mm² vers un nombre de pixels selon A4/A3 ;
7. fusionne les petites régions avec une stratégie configurable ;
8. exporte les cartes, contours, graphes et métriques avant/après.

La numérotation, le SVG et le PDF appartiennent au Lot 3.

## Prérequis

- Python 3.11 ou 3.12 ;
- Windows, macOS ou Linux ;
- environ 1 Go de mémoire pour une image de 1 200 pixels de côté.

## Installation

Depuis le dossier du projet :

```bash
python -m venv .venv
```

Activation :

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Installation :

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Premier test

```bash
coloriage-lot2 examples/demo.png \
  --output results/demo \
  --colors 8
```

Sans installation du raccourci :

```bash
PYTHONPATH=src python -m coloriage_lot2 examples/demo.png \
  --output results/demo \
  --colors 8
```

Sous Windows PowerShell :

```powershell
$env:PYTHONPATH = "src"
python -m coloriage_lot2 examples/demo.png --output results/demo --colors 8
```

## Comparer les trois stratégies de fusion

```bash
coloriage-lot2 examples/demo.png \
  --output results/comparaison \
  --colors 8 \
  --compare-strategies
```

Trois sous-dossiers sont créés :

- `color` privilégie la couleur voisine la plus proche ;
- `boundary` privilégie la frontière partagée la plus longue ;
- `balanced` cherche un compromis entre fidélité colorimétrique et forme.

La racine contient `strategies-comparison.png`, `.csv` et `.json`.

## Tester une photo réelle

Réglage standard A4 :

```bash
coloriage-lot2 "/chemin/vers/photo.jpg" \
  --output results/photo-a4 \
  --colors 12 \
  --page-format a4 \
  --orientation portrait \
  --min-region-area-mm2 9 \
  --superpixels 900
```

Comparer le même rendu en A3 :

```bash
coloriage-lot2 "/chemin/vers/photo.jpg" \
  --output results/photo-a3 \
  --colors 12 \
  --page-format a3 \
  --orientation portrait \
  --min-region-area-mm2 9
```

À surface physique identique, une image A3 conserve davantage de petites zones
qu’une image A4, car chaque pixel occupe plus de place sur le papier.

## Paramètres importants

| Option | Défaut | Rôle |
|---|---:|---|
| `--colors` | 12 | Nombre de couleurs, de 2 à 40 |
| `--segmentation` | `slic` | `slic` ou baseline `components` |
| `--superpixels` | 900 | Niveau de détail spatial de SLIC |
| `--compactness` | 10 | Régularité des superpixels |
| `--smoothing-radius` | 1 px | Lissage local des contours |
| `--merge-strategy` | `balanced` | `color`, `boundary` ou `balanced` |
| `--color-tolerance` | 35 | Delta E au-delà duquel une fusion est signalée |
| `--page-format` | `a4` | `a4` ou `a3` |
| `--orientation` | `portrait` | `portrait` ou `landscape` |
| `--margin-mm` | 12 mm | Marge autour de l’image |
| `--min-region-area-mm2` | 9 mm² | Surface minimale d’une région |
| `--thin-width-mm` | 1,5 mm | Seuil de diagnostic des zones étroites |

Repères de départ pour la surface minimale :

| Difficulté | Surface minimale suggérée |
|---|---:|
| Simple | 16 à 25 mm² |
| Standard | 9 mm² |
| Détaillée | 4 à 6 mm² |

Ces valeurs doivent être validées par des impressions réelles.

## Fichiers produits

| Fichier | Contenu |
|---|---|
| `overview.png` | Comparaison source, palette, avant et après fusion |
| `segmented-before.png` | Couleurs après segmentation et lissage |
| `merged-after.png` | Couleurs après fusion |
| `model-before.png` / `model-after.png` | Modèles colorés avec frontières |
| `contours-before.png` / `contours-after.png` | Contours noirs sans numéros |
| `regions-*-preview.png` | Identifiants de régions en fausses couleurs |
| `region-labels-*.npy` | Matrices exactes des identifiants |
| `regions-before.csv` / `regions-after.csv` | Géométrie et métriques physiques |
| `adjacency-before.csv` / `adjacency-after.csv` | Graphe de voisinage |
| `merges.csv` | Journal de toutes les fusions |
| `palette.csv` | Palette RGB, Lab et occupation avant/après |
| `stats.json` | Paramètres, métriques, géométrie papier et temps |

## Lire le diagnostic

Dans `stats.json`, surveiller :

- `regions_before.count` et `regions_after.count` ;
- `below_area_threshold_count`, idéalement égal à zéro ;
- `thin_region_count`, qui signale les zones dont l’épaisseur maximale est
  inférieure au seuil ;
- `recolored_pixel_percent`, pour vérifier que la fusion ne modifie pas une
  part excessive de l’image ;
- `forced_merges_above_color_tolerance`, qui révèle les fusions entre couleurs
  éloignées ;
- `total` dans `timings_ms`, avec une cible de preview sous cinq secondes.

Une région peut dépasser la surface minimale tout en restant longue et très
fine. Le Lot 2 la signale mais ne la transforme pas encore en trait : cette
décision sera exploitée lors de la génération du coloriage au Lot 3.

## Protocole de validation

Tester idéalement 5 à 10 images :

- portrait ;
- animal ;
- paysage ;
- objet sur fond simple ;
- scène avec arrière-plan chargé.

Pour chaque image :

1. lancer 8, 12 et 16 couleurs ;
2. comparer `components` et `slic` ;
3. lancer `--compare-strategies` sur le meilleur réglage ;
4. examiner `overview.png` et `contours-after.png` ;
5. noter la fidélité du sujet et les détails importants perdus.

Les fichiers les plus utiles à transmettre pour l’analyse sont :

- `overview.png` ;
- `contours-after.png` ;
- `stats.json` ;
- `strategies-comparison.png` si le mode comparaison a été utilisé.

## Tests automatisés

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Limites connues

- SLIC et K-means ne comprennent pas le sujet de la photo.
- Le Delta E utilisé pour le graphe est la distance CIE76, suffisante pour le
  POC mais améliorable.
- Les zones fines sont diagnostiquées, pas encore converties en traits.
- Les contours sont des aperçus bitmap ; le tracé vectoriel arrive au Lot 3.
- Le seuil de 9 mm² est une hypothèse à valider sur papier.
