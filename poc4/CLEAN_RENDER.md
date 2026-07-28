# Clean Render v1

Cette migration améliore rapidement la qualité du coloriage sans remplacer le
moteur déterministe par un modèle génératif.

## Changements

- prétraitement bilatéral préservant les contours ;
- SLICO (`scikit-image`) avec connectivité garantie ;
- fusion des couleurs trop proches avec Delta E 2000 ;
- passages supplémentaires de fusion pour les bandes trop fines ;
- contours de remplissage subpixel ;
- frontières partagées tracées une seule fois ;
- calque de détails internes multiscale ;
- simplification exprimée en millimètres imprimés ;
- conservation du mode historique `slic_legacy` pour les comparaisons.

## Installation

```bash
cd poc4
python -m pip install -e ".[ai]"
```

## Vérification

```bash
PYTHONPATH=poc4/src python -m unittest discover -s poc4/tests -v
npm run lint
npm test
```

## Comparaison recommandée

Générer les mêmes photos avec `slic` et `slic_legacy`, puis comparer :

- continuité des silhouettes ;
- fragmentation des visages ;
- nombre de régions après fusion ;
- nombre de polices réduites sous le seuil ;
- lisibilité sur une impression A4 réelle.
