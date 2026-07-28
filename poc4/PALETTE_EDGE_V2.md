# Palette globale v2 et fusion guidée par contours

Cette migration introduit deux évolutions compatibles avec le pipeline v1.

## Palette globale

`palette_mode` accepte :

- `legacy` : comportement comparable à la v1 ;
- `adaptive` : le nombre demandé est un maximum et les doublons perceptuels sont supprimés globalement ;
- `exact` : les doublons sont supprimés puis des couleurs utiles sont réintroduites pour atteindre le nombre demandé lorsque l'image le permet.

En mode sujet, la palette est désormais partagée entre le sujet et le fond. La séparation spatiale est toujours pilotée par le masque, mais deux couleurs identiques ne reçoivent plus deux numéros différents.

## Fusion guidée par contours

Une carte de contours combine luminance, chrominance, Canny, silhouette du sujet et limites des zones détaillées. La fusion :

- pénalise les frontières visuellement fortes ;
- évite une frontière protégée lorsqu'une autre voisine est disponible ;
- conserve un fallback afin qu'une micro-zone ne reste jamais bloquée sans numéro.

## Comparaison

```bash
PYTHONPATH=poc4/src python -m coloriage_lot3.benchmark   --output benchmark-output   --assert-quality
```

Le benchmark compare maintenant `legacy`, `v1` et `v2`.
