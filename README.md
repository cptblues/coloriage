# Nuance — POC de coloriage mystère

Interface web du POC Nuance. Elle permet d’importer une photo, de contrôler
l’isolation locale du sujet avec `rembg`, de tracer des zones où conserver
davantage de détails, de régler le rendu et de télécharger un aperçu.

## Fonctionnalités

- import local de fichiers JPG, JPEG et PNG ;
- contrôle visuel du masque sujet généré localement ;
- tracé de zones détaillées à la souris ou au tactile ;
- réglage de 8 à 40 couleurs ;
- choix A4/A3, orientation et niveau de détail ;
- aperçu réel du coloriage et du modèle couleur générés par le moteur Python ;
- téléchargement PNG par page et PDF complet ;
- interface responsive.

En mode local, la photo reste sur votre machine. En mode hébergé, elle est
envoyée à votre propre moteur Python, pas à une API tierce.

## Lancer le projet

Prérequis :

- Node.js 22.13 ou plus récent ;
- Python 3.11 ou plus récent ;
- dépendances Python du POC, avec `rembg` pour l’isolation IA.

```bash
npm install
cd poc4
python -m pip install -e ".[ai]"
cd ..
```

Lancez le moteur Python local dans un terminal :

```bash
npm run dev:engine
```

Lancez l’interface web dans un deuxième terminal :

```bash
npm run dev
```

Ouvrez ensuite l’adresse indiquée dans le terminal.

Par défaut, le front cherche le moteur sur `http://127.0.0.1:8765`. Pour une
autre URL, créez un fichier `.env.local` :

```bash
VITE_ENGINE_URL=http://adresse-du-moteur:8765
```

## Héberger avec Docker Compose

Le projet contient un packaging simple pour un serveur Linux avec Docker :

```bash
cp .env.example .env
docker compose up --build
```

L’application est alors disponible sur le port `COLORIAGE_PUBLIC_PORT`, `8080`
par défaut. Caddy expose le front et relaie `/engine/*` vers le moteur Python.

Services :

- `frontend` : build et sert l’interface Vinext ;
- `engine` : lance le moteur Python avec `rembg` ;
- `caddy` : reverse proxy public.

Au premier usage du détourage IA, le moteur télécharge le modèle `rembg` dans
le volume Docker `rembg-models`, puis le réutilise.

Variables principales dans `.env` :

```bash
COLORIAGE_PUBLIC_PORT=8080
VITE_ENGINE_URL=/engine
COLORIAGE_MAX_IMAGE_MB=40
COLORIAGE_MAX_REQUEST_MB=64
COLORIAGE_DEFAULT_MAX_SIDE=1200
COLORIAGE_MAX_SIDE_LIMIT=2400
```

Le front envoie `maxSide: "auto"` : le moteur utilise environ `900 px` pour
les petites images, `COLORIAGE_DEFAULT_MAX_SIDE` pour le cas standard, et
jusqu’à `1600 px` pour les grandes photos si `COLORIAGE_MAX_SIDE_LIMIT`
l’autorise.

## Tracer une zone détaillée

À l’étape « Zones détaillées » :

1. choisissez « Pinceau » pour peindre directement la zone, ou « Contour »
   pour entourer une zone qui sera remplie ;
2. ajustez l’épaisseur ;
3. maintenez le bouton de la souris ou le doigt ;
4. relâchez pour enregistrer la zone.

Le bouton « Annuler » retire la dernière zone et « Tout effacer » recommence
la sélection.

## Vérifier le projet

```bash
npm run lint
npm test
```

## Mettre le code sur GitHub

Décompressez l’archive, puis exécutez :

```bash
git init
git add .
git commit -m "Initial Nuance POC"
git branch -M main
git remote add origin https://github.com/VOTRE-COMPTE/VOTRE-DEPOT.git
git push -u origin main
```

Ne versionnez pas `node_modules`, `dist` ou les fichiers locaux
d’environnement ; ils sont déjà couverts par `.gitignore`.

Ce projet utilise Next.js avec Vinext. Il peut être stocké sur GitHub, mais il
ne s’exécute pas directement sur GitHub Pages sans adaptation en site statique.
Pour le mettre en ligne tel quel, utilisez le packaging Docker Compose fourni
ou un hébergement compatible Next.js/Cloudflare Workers avec un moteur Python
séparé.

## Structure utile

```text
app/page.tsx       parcours et interactions
app/globals.css    direction visuelle « Atelier ludique »
public/            images et ressources statiques
tests/             vérifications automatisées
```
