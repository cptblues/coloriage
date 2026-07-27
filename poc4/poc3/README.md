# POC Coloriage mystère — Lot 3

Ce lot transforme les régions validées au Lot 2 en une première feuille de
coloriage exploitable :

1. import local d’une image **JPG, JPEG ou PNG** ;
2. normalisation, quantification CIELAB et segmentation ;
3. fusion des micro-régions selon le format A4/A3 ;
4. placement des numéros au point le plus éloigné des contours ;
5. réduction contrôlée de la taille des numéros ;
6. signalement des régions où aucun numéro lisible ne tient ;
7. génération d’une feuille numérotée et d’un modèle coloré en SVG ;
8. ajout d’une légende numérotée avec codes hexadécimaux.

Le traitement s’effectue entièrement sur votre ordinateur. La photo n’est
envoyée vers aucun service.

## Prérequis

- Python 3.11 ou 3.12 ;
- Windows, macOS ou Linux ;
- environ 1 Go de mémoire pour une image de 1 200 pixels de côté.

## Installation

Depuis le dossier extrait :

```bash
python -m venv .venv
```

Activez l’environnement :

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Puis installez le projet :

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Test immédiat

```bash
coloriage-lot3 examples/demo.png \
  --output results/demo \
  --colors 8
```

Sans installation du raccourci :

```bash
PYTHONPATH=src python -m coloriage_lot3 examples/demo.png \
  --output results/demo \
  --colors 8
```

## Tester votre photo JPG

Placez par exemple `ma-photo.jpg` dans le dossier du projet, puis lancez :

```bash
coloriage-lot3 "ma-photo.jpg" \
  --output results/ma-photo-12 \
  --colors 12 \
  --page-format a4 \
  --orientation portrait \
  --title "Mon coloriage"
```

Les chemins contenant des espaces doivent rester entre guillemets. Les
extensions `.jpg`, `.jpeg` et `.png` sont acceptées. L’orientation EXIF des
photos de téléphone est corrigée automatiquement.

Pour comparer les niveaux de détail :

```bash
coloriage-lot3 "ma-photo.jpg" --output results/photo-08 --colors 8
coloriage-lot3 "ma-photo.jpg" --output results/photo-12 --colors 12
coloriage-lot3 "ma-photo.jpg" --output results/photo-16 --colors 16
```

## Résultats à ouvrir en premier

| Fichier | Utilité |
|---|---|
| `apercu-coloriage.png` | Aperçu rapide de la feuille numérotée |
| `coloriage.svg` | Feuille vectorielle A4/A3 à ouvrir ou imprimer |
| `apercu-modele-couleur.png` | Aperçu du modèle coloré |
| `modele-couleur.svg` | Modèle vectoriel avec palette |
| `overview.png` | Diagnostic source, palette et fusion |
| `placements-numeros.csv` | Position et statut de chaque numéro |
| `stats.json` | Mesures complètes du traitement |

Les SVG s’ouvrent dans un navigateur récent, Inkscape, Illustrator ou Affinity
Designer. Ils peuvent être imprimés directement ou convertis en PDF par ces
outils.

## Paramètres du coloriage

| Option | Défaut | Rôle |
|---|---:|---|
| `--colors` | 12 | Nombre de couleurs, de 2 à 40 |
| `--superpixels` | 900 | Niveau de détail spatial |
| `--min-region-area-mm2` | 9 mm² | Surface minimale d’une région |
| `--number-font-mm` | 2,8 mm | Taille préférée des numéros |
| `--min-number-font-mm` | 1,8 mm | Limite de réduction avant exclusion |
| `--number-padding-mm` | 0,45 mm | Blanc autour du numéro |
| `--line-width-mm` | 0,25 mm | Épaisseur des contours |
| `--page-format` | `a4` | `a4` ou `a3` |
| `--orientation` | `portrait` | `portrait` ou `landscape` |
| `--title` | `Coloriage mystère` | Titre imprimé |

Repères de difficulté :

| Niveau | Couleurs | Surface minimale |
|---|---:|---:|
| Simple | 6 à 10 | 16 à 25 mm² |
| Standard | 10 à 16 | 9 mm² |
| Détaillé | 16 à 24 | 4 à 6 mm² |

## Vérifier la numérotation

La sortie console et `stats.json` donnent :

- `numbers_placed` / `placed_count` : numéros réellement placés ;
- `numbers_skipped` / `skipped_count` : régions trop étroites ;
- `coverage_percent` : couverture de numérotation ;
- `skipped_region_ids` : régions à examiner.

Dans `placements-numeros.csv`, une ligne `status=skipped` avec
`reason=zone_trop_etroite` signifie que le moteur a préféré ne rien imprimer
plutôt que de produire un numéro illisible.

Si trop de numéros sont ignorés :

1. augmentez `--min-region-area-mm2` ;
2. diminuez `--superpixels` ;
3. réduisez légèrement `--min-number-font-mm` sans descendre sous une taille
   lisible à l’impression ;
4. essayez le format A3.

## Tester les trois stratégies de fusion

```bash
coloriage-lot3 "ma-photo.jpg" \
  --output results/comparaison \
  --colors 12 \
  --compare-strategies
```

Les sous-dossiers `color`, `boundary` et `balanced` contiennent chacun leurs
SVG et leurs aperçus.

## Tests automatisés

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Les tests couvrent notamment l’import JPEG, la génération de SVG XML valides,
le placement d’un numéro et le refus d’une zone trop étroite.

## Limites du Lot 3

- Les contours SVG suivent encore la grille de segmentation et peuvent paraître
  anguleux à fort zoom.
- Le moteur ne comprend pas encore le sujet d’une photographie : il ne protège
  pas spécifiquement les yeux ou les traits d’un visage.
- Une région trop étroite est signalée, mais aucun trait de rappel externe
  n’est encore créé.
- Le SVG est prêt pour les essais ; l’export PDF prépresse sera traité dans un
  lot ultérieur.

## Retour de test utile

Pour une photo personnelle, transmettez seulement si vous le souhaitez :

- `apercu-coloriage.png` ;
- `apercu-modele-couleur.png` ;
- `overview.png` ;
- `stats.json`.

La photo originale n’est pas nécessaire pour diagnostiquer l’installation,
mais elle aide à juger la fidélité du résultat.
