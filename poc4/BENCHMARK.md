# Benchmark visuel

Le benchmark compare `slic_legacy` et le profil `v1` sur trois images synthétiques déterministes.

```bash
PYTHONPATH=poc4/src python -m coloriage_lot3.benchmark \
  --output benchmark-output \
  --assert-quality
```

La commande génère les aperçus, une comparaison côte à côte et un rapport JSON.
Elle échoue si un numéro est ignoré, si la couverture descend sous 100 %, ou si
la palette/région produite est invalide.
