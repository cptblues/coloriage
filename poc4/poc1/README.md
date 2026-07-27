# POC Coloriage mystère — Lot 1

Baseline locale du moteur de transformation d’image.

Ce projet :

1. lit une image JPG ou PNG et corrige son orientation EXIF ;
2. convertit l’image en RGB/sRGB et la redimensionne ;
3. réduit sa palette en espace perceptuel CIELAB avec K-means ;
4. détecte les composantes connexes de chaque couleur ;
5. exporte les images, la palette, la carte des régions et les statistiques.

Le Lot 1 ne fusionne pas encore les micro-zones, ne dessine pas les contours et
ne place pas de numéros. Ces étapes correspondent aux Lots 2 et 3.

## Prérequis

- Python 3.11 ou 3.12 ;
- environ 1 Go de mémoire pour les images de démonstration ;
- Windows, macOS ou Linux.

## Installation

Depuis le dossier du projet :

```bash
python -m venv .venv
```

Activation de l’environnement :

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

Une image synthétique libre de droits est fournie dans `examples/demo.png`.

```bash
coloriage-lot1 examples/demo.png --output results/demo --colors 8
```

Sans installation du raccourci de commande :

```bash
PYTHONPATH=src python -m coloriage_lot1 examples/demo.png \
  --output results/demo --colors 8
```

Sous Windows PowerShell, remplacer temporairement `PYTHONPATH=src` par :

```powershell
$env:PYTHONPATH = "src"
python -m coloriage_lot1 examples/demo.png --output results/demo --colors 8
```

## Tester une photo

```bash
coloriage-lot1 "/chemin/vers/photo.jpg" \
  --output results/photo-12-couleurs \
  --colors 12 \
  --max-side 1200 \
  --connectivity 8 \
  --seed 42
```

Paramètres principaux :

| Option | Valeur par défaut | Rôle |
|---|---:|---|
| `--colors` | 12 | Nombre demandé de couleurs, entre 2 et 40 |
| `--max-side` | 1200 | Taille maximale du plus grand côté |
| `--sample-pixels` | 100000 | Pixels utilisés pour entraîner K-means |
| `--connectivity` | 8 | Voisinage 4 ou 8 pour les régions |
| `--seed` | 42 | Résultat reproductible |
| `--overwrite` | désactivé | Autorise l’écrasement d’un dossier non vide |

Pour une première comparaison, exécuter la même photo avec 8, 12, 16, 20 et
24 couleurs dans des dossiers distincts.

## Fichiers produits

| Fichier | Contenu |
|---|---|
| `normalized.png` | Image corrigée et redimensionnée utilisée par le moteur |
| `quantized.png` | Image limitée à la palette calculée |
| `regions-preview.png` | Visualisation où chaque région a une couleur arbitraire |
| `region-labels.npy` | Matrice exacte des identifiants de région |
| `palette.csv` | Palette RGB/hex/Lab, fréquence et nombre de régions |
| `regions.csv` | Une ligne par composante connexe avec surface et boîte |
| `stats.json` | Paramètres, temps de calcul et métriques agrégées |
| `overview.png` | Planche de comparaison rapide |

`regions-preview.png` ne représente pas les couleurs finales. Ses couleurs
servent uniquement à distinguer deux composantes voisines.

## Lire les métriques

Le nombre de couleurs est différent du nombre de régions. Une image à
12 couleurs peut encore contenir plusieurs milliers de composantes.

Les signaux à surveiller :

- beaucoup de régions de 1 à 10 pixels : bruit ou texture trop détaillée ;
- un nombre de régions très supérieur à ce qui peut être colorié ;
- une palette dont plusieurs teintes sont presque identiques ;
- une image quantifiée qui perd les traits importants du sujet.

Le Lot 2 devra utiliser ces mesures pour fusionner les petites régions et
construire des zones réellement coloriables.

## Tests

Les tests utilisent uniquement `unittest`, inclus avec Python :

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Arborescence

```text
coloriage-poc-lot1/
├── examples/
│   └── demo.png
├── scripts/
│   └── create_demo_image.py
├── src/coloriage_lot1/
│   ├── cli.py
│   ├── color.py
│   ├── export.py
│   ├── pipeline.py
│   └── regions.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Limites connues

- K-means traite uniquement les couleurs, sans compréhension du sujet.
- Une texture fine crée beaucoup de petites composantes.
- Les régions ne sont pas encore fusionnées ni lissées.
- La palette est optimisée pour l’écran, pas pour une marque de crayons.
- Aucun SVG, PDF, contour ou numéro n’est produit dans ce lot.

