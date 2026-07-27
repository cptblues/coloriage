# POC Coloriage mystère — Lot 3.1 IA locale

Ce lot ajoute une compréhension minimale de l’image au moteur déterministe du
Lot 3. Une IA locale produit uniquement un **masque du sujet**. La palette, les
régions, les contours, les numéros et les SVG restent calculés par le moteur
classique.

Le sujet et le fond reçoivent des réglages différents :

| Réglage par défaut | Sujet | Fond |
|---|---:|---:|
| Part des couleurs | 68 % | 32 % |
| Surface minimale | 6 mm² | 28 mm² |
| Détail spatial | 100 % | 45 % |
| Lissage | léger | renforcé |

Le traitement est local. Aucune photo n’est envoyée à une API.

## Prérequis

- Python 3.11, 3.12 ou 3.13 ;
- Windows, macOS ou Linux ;
- 2 à 4 Go de mémoire recommandés pour le mode IA ;
- une connexion au premier lancement pour télécharger le modèle.

## Installation sans IA

Cette installation permet d’utiliser un masque manuel et d’exécuter les tests :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Sous Windows PowerShell, l’activation est :

```powershell
.venv\Scripts\Activate.ps1
```

## Installation avec IA locale sur CPU

```bash
python -m pip install -e ".[ai]"
```

`rembg` télécharge automatiquement les poids du modèle au premier usage. Ils
sont ensuite réutilisés localement. Le premier lancement est donc nettement
plus long que les suivants.

## Test recommandé sur votre JPG

```bash
coloriage-lot31 "ma-photo.jpg" \
  --output results/ma-photo-ia \
  --colors 24 \
  --subject-mode ai \
  --ai-model birefnet-general \
  --page-format a4 \
  --orientation portrait \
  --title "Mon coloriage"
```

Pour un portrait humain, comparez avec :

```bash
coloriage-lot31 "ma-photo.jpg" \
  --output results/ma-photo-portrait \
  --colors 24 \
  --subject-mode ai \
  --ai-model birefnet-portrait
```

Le modèle `birefnet-general-lite` est une alternative plus légère pour une
machine lente.

## Test immédiat sans modèle

Le projet contient une illustration et un masque de démonstration :

```bash
coloriage-lot31 examples/demo.png \
  --output results/demo-sujet \
  --colors 8 \
  --subject-mask examples/demo-mask.png
```

Un résultat déjà généré se trouve dans `examples/result-reference-subject/`.

## Tester sans télécharger l’IA

Créez un PNG noir et blanc de la même image : blanc pour le sujet, noir pour le
fond. Le masque peut avoir une autre taille ; il sera redimensionné.

```bash
coloriage-lot31 "ma-photo.jpg" \
  --output results/ma-photo-masque \
  --colors 24 \
  --subject-mask "mon-masque.png"
```

Le masque manuel est aussi le mécanisme de correction lorsqu’un détourage IA
est imparfait : retouchez `masque-sujet.png`, puis relancez avec
`--subject-mask`.

## Résultats à regarder

| Fichier | Utilité |
|---|---|
| `controle-masque.png` | Vérifier immédiatement le détourage |
| `masque-sujet.png` | Masque réutilisable et retouchable |
| `comparaison-sujet-fond.png` | Photo, masque et simplification côte à côte |
| `apercu-coloriage.png` | Feuille numérotée |
| `apercu-modele-couleur.png` | Modèle coloré |
| `palette-page.png` | Palette sur page séparée si `--palette-layout separate` |
| `coloriage.pdf` | PDF deux pages : coloriage puis palette |
| `coloriage.svg` | Feuille vectorielle A4/A3 |
| `modele-couleur.svg` | Modèle vectoriel |
| `stats.json` | Mesures globales et séparées sujet/fond |

La silhouette du sujet est tracée 1,8 fois plus épaisse que les contours
internes.

## Serveur local et production

Le serveur HTTP du moteur expose `/health`, `/mask` et `/generate` :

```bash
PYTHONPATH=src python -m coloriage_lot3.server
```

Variables d’environnement prises en charge :

```bash
COLORIAGE_ENGINE_HOST=0.0.0.0
COLORIAGE_ENGINE_PORT=8765
COLORIAGE_MAX_IMAGE_MB=40
COLORIAGE_MAX_REQUEST_MB=64
COLORIAGE_DEFAULT_MAX_SIDE=1200
COLORIAGE_MAX_SIDE_LIMIT=2400
COLORIAGE_MAX_DETAIL_ZONES=64
COLORIAGE_MAX_DETAIL_POINTS=12000
```

Si `maxSide` est omis ou vaut `"auto"`, le serveur choisit la résolution
interne selon la photo : petite image plus douce, cas standard à
`COLORIAGE_DEFAULT_MAX_SIDE`, grande photo jusqu’à `1600 px` si la limite
l’autorise. Une valeur numérique explicite reste supportée.

Le payload `/generate` renvoie aussi `pdfDocument`, un data URL
`application/pdf` prêt à télécharger.

## Réglages utiles

```text
--subject-color-ratio 0.68
--subject-min-region-area-mm2 6
--background-min-region-area-mm2 28
--background-superpixel-ratio 0.45
--background-smoothing-radius 3
--mask-threshold 128
```

Si le visage perd trop de détails, augmentez `--subject-color-ratio` vers
`0.72`, baissez `--subject-min-region-area-mm2` vers `4`, ou utilisez 24 à
28 couleurs.

Si le décor reste trop présent, augmentez
`--background-min-region-area-mm2` vers `35` ou baissez
`--background-superpixel-ratio` vers `0.30`.

## Comparer au Lot 3 sans sujet prioritaire

```bash
coloriage-lot31 "ma-photo.jpg" \
  --output results/sans-ia \
  --colors 24

coloriage-lot31 "ma-photo.jpg" \
  --output results/avec-ia \
  --colors 24 \
  --subject-mode ai
```

## Tests automatisés

Les tests n’ont pas besoin de télécharger un modèle :

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Ils couvrent notamment le masque manuel, l’allocation distincte de la palette,
les seuils sujet/fond, l’export des diagnostics et le message d’installation
du mode IA.

## Limites du POC

- BiRefNet peut sélectionner le mauvais sujet dans une scène contenant
  plusieurs personnes ou animaux.
- Les cheveux, moustaches et objets transparents peuvent produire des bordures
  imparfaites.
- Le POC ne contient pas encore d’éditeur de masque interactif.
- Les détails du visage ne sont pas détectés séparément : ils bénéficient de la
  palette du sujet, mais ne sont pas encore convertis en traits noirs.
- Le téléchargement et l’inférence IA dépendent de `rembg` et ONNX Runtime ;
  le reste du moteur demeure utilisable sans ces dépendances.

## Retour de test utile

Transmettez, si vous le souhaitez :

- `controle-masque.png` ;
- `comparaison-sujet-fond.png` ;
- `apercu-coloriage.png` ;
- `apercu-modele-couleur.png` ;
- `stats.json`.

La photo originale n’est pas obligatoire pour diagnostiquer le pipeline.
