# Projet d’application de coloriage mystère

## 1. Résumé du projet

L’objectif est de créer une application qui transforme une photo importée par un utilisateur en un coloriage mystère imprimable :

1. l’utilisateur charge une photo ;
2. il choisit le format et recadre l’image ;
3. le moteur simplifie la photo en zones de couleurs ;
4. l’utilisateur ajuste le nombre de couleurs et le niveau de détail dans une prévisualisation ;
5. l’application génère un coloriage numéroté, son modèle coloré et sa palette ;
6. l’utilisateur télécharge un PDF ou commande une impression premium.

La difficulté principale n’est pas l’interface ou le paiement, mais la transformation automatique d’une photo en un dessin :

- fidèle au sujet ;
- agréable visuellement ;
- constitué de zones assez grandes pour être coloriées ;
- doté de contours propres et de numéros lisibles ;
- compatible avec une impression A4 ou A3.

## 2. Recommandation produit

### Plateforme initiale

Commencer par une **application web responsive et installable (PWA)**.

Ce choix permet :

- un accès immédiat sur ordinateur, tablette et mobile ;
- des mises à jour centralisées ;
- un parcours simple pour le paiement et la commande ;
- une acquisition plus facile depuis les moteurs de recherche et les réseaux sociaux ;
- la réutilisation ultérieure du frontend dans une application desktop avec Tauri.

Une version desktop ne devient prioritaire que si les utilisateurs demandent :

- un fonctionnement hors ligne ;
- un traitement exclusivement local pour la confidentialité ;
- le traitement de très grandes images ;
- une licence professionnelle indépendante du service en ligne.

### Positionnement

Le service peut viser plusieurs usages :

- cadeau personnalisé ;
- loisir créatif ;
- portrait de famille ou d’animal ;
- activité pour enfants ou seniors ;
- produit destiné aux photographes, écoles, maisons de retraite ou créateurs.

## 3. Parcours utilisateur cible

### 3.1 Import

Formats initiaux :

- JPG ;
- PNG ;
- éventuellement HEIC dans une version ultérieure.

Contrôles automatiques :

- résolution ;
- orientation EXIF ;
- netteté ;
- contraste ;
- niveau de détail ;
- dimensions compatibles avec le format choisi.

L’application peut recommander des paramètres :

> Cette image contient beaucoup de détails. Le réglage « Loisir », avec 18 couleurs et une simplification moyenne, est recommandé.

### 3.2 Choix du format et recadrage

À terme :

- A4 portrait ;
- A4 paysage ;
- A3 portrait ;
- A3 paysage.

Pour le MVP, A4 portrait et paysage suffisent. L’utilisateur doit pouvoir zoomer, déplacer et recadrer la photo dans la zone imprimable.

### 3.3 Prévisualisation et réglages

Trois vues sont utiles :

- photo originale ;
- version simplifiée et colorée ;
- coloriage avec contours et numéros.

Une quatrième vue peut montrer le résultat final colorié.

Réglages principaux :

- nombre de couleurs ;
- niveau de détail ;
- taille minimale des zones ;
- épaisseur des contours ;
- taille des numéros ;
- traitement de l’arrière-plan ;
- format et orientation.

### 3.4 Validation et achat

Deux produits :

- téléchargement d’un pack PDF ;
- tirage premium expédié à domicile.

Le pack PDF peut contenir :

1. le coloriage numéroté ;
2. le modèle coloré ;
3. la palette et quelques conseils.

## 4. Moteur de transformation d’image

Le moteur doit être déterministe : une même image avec les mêmes paramètres doit produire le même résultat.

### 4.1 Pipeline recommandé

#### Étape A — Normalisation

- correction de l’orientation ;
- conversion dans un profil colorimétrique maîtrisé ;
- redimensionnement ;
- légère réduction du bruit ;
- correction optionnelle du contraste.

#### Étape B — Simplification

Appliquer un filtre préservant les contours, par exemple :

- filtre bilatéral ;
- mean-shift ;
- diffusion anisotrope.

Le but est de supprimer les textures inutiles sans faire disparaître les éléments importants comme les yeux, le museau ou les contours d’un visage.

#### Étape C — Réduction de la palette

Convertir les couleurs en espace CIELAB, puis les regrouper avec :

- K-means pour le premier prototype ;
- éventuellement median cut ou octree plus tard ;
- une distance perceptuelle pour éviter des couleurs presque indiscernables.

Le **nombre de couleurs** et le **nombre de zones** doivent rester deux paramètres distincts. Une image de dix couleurs peut encore contenir des milliers de petites régions.

#### Étape D — Segmentation

Approche initiale recommandée :

1. création de superpixels SLIC ;
2. attribution d’une couleur de palette aux superpixels ;
3. construction des relations de voisinage ;
4. fusion des régions voisines proches ;
5. suppression ou fusion des composantes trop petites ;
6. lissage et simplification des contours.

Une approche plus simple par quantification puis composantes connexes peut aussi servir de baseline pour le POC.

#### Étape E — Graphe de régions

Chaque région contient au minimum :

- son identifiant ;
- sa couleur et son numéro de palette ;
- sa surface ;
- son contour ;
- ses régions voisines ;
- sa boîte englobante ;
- un emplacement possible pour son numéro.

Ce graphe permet de choisir intelligemment la région voisine lors d’une fusion.

#### Étape F — Placement des numéros

Le numéro doit :

- rester entièrement dans la région quand cela est possible ;
- ne pas chevaucher un contour ;
- être lisible à la taille physique d’impression ;
- ne pas recouvrir un autre numéro.

Une première méthode peut utiliser le maximum de la transformée de distance de la région. Les zones trop petites ou trop étroites doivent être :

- fusionnées ;
- supprimées si elles ne sont pas importantes ;
- ou, plus tard, associées à un numéro externe avec une ligne de rappel.

#### Étape G — Vectorisation et export

Le format de travail final recommandé est le SVG :

- contours vectoriels ;
- numéros ;
- palette ;
- marges ;
- éléments de mise en page.

Le SVG peut ensuite être converti en PDF pour :

- le téléchargement domestique ;
- l’impression professionnelle.

## 5. Réglages et préréglages

### Réglages visibles

| Réglage | Effet |
|---|---|
| Nombre de couleurs | Taille de la palette |
| Difficulté | Niveau global de simplification |
| Niveau de détail | Nombre de zones et conservation des petits objets |
| Taille minimale des zones | Élimination des zones impossibles à colorier |
| Épaisseur des lignes | Lisibilité du dessin imprimé |
| Arrière-plan | Conservation, forte simplification ou suppression |

### Préréglages proposés

| Préréglage | Couleurs indicatives | Caractéristiques |
|---|---:|---|
| Enfant | 6 à 10 | Grandes zones, contours renforcés |
| Loisir | 11 à 18 | Équilibre entre fidélité et facilité |
| Détaillé | 19 à 24 | Plus de nuances et de petites zones |
| Expert | 25 à 40 | À réserver aux grands formats et aux images adaptées |

Les paramètres techniques avancés doivent rester repliés par défaut.

## 6. Formats et impression

### Dimensions

- A4 : 210 × 297 mm ;
- A3 : 297 × 420 mm.

Les seuils de taille des zones et des numéros doivent être exprimés ou vérifiés en dimensions physiques, pas uniquement en pixels. Une région acceptable en A3 peut devenir illisible en A4.

### PDF utilisateur

Le PDF téléchargeable doit prévoir :

- des marges sûres pour les imprimantes domestiques ;
- des polices incorporées ou des chiffres vectorisés ;
- un coloriage en traits noirs ;
- un modèle coloré ;
- une palette numérotée.

### Fichier d’impression

Selon les exigences du partenaire :

- PDF/X-4 ou PDF/X-1a ;
- profil colorimétrique adapté ;
- fonds perdus ;
- traits de coupe ;
- éléments vectoriels ;
- résolution suffisante des éventuelles images bitmap.

### Papier premium

Le terme doit correspondre à une spécification mesurable, par exemple :

> Papier blanc naturel non couché, 200 g/m², adapté aux crayons de couleur et aux feutres à base d’eau.

Des échantillons doivent être testés avec :

- crayons ;
- feutres à eau ;
- marqueurs ;
- gomme ;
- éventuellement feutres à alcool, avec vérification de la traversée de l’encre.

## 7. Architecture technique cible

### Frontend

- React ou Vue ;
- TypeScript ;
- Canvas et/ou SVG pour la prévisualisation ;
- composant de recadrage ;
- Web Workers pour les traitements locaux ;
- PWA.

### Backend

- Python ;
- FastAPI ;
- OpenCV ;
- scikit-image ;
- NumPy ;
- scikit-learn ;
- bibliothèque de génération SVG/PDF.

### Infrastructure commerciale

- PostgreSQL pour utilisateurs, projets, paramètres et commandes ;
- stockage objet compatible S3 pour les originaux et exports ;
- Redis et une file de tâches pour les traitements longs ;
- workers de génération séparés de l’API ;
- Stripe Checkout et webhooks ;
- API du partenaire d’impression.

### Flux cible

```text
Import et recadrage
        ↓
Prévisualisation basse résolution
        ↓
Validation des paramètres
        ↓
Tâche de génération haute définition
        ↓
SVG + PDF + modèle coloré
        ↓
Téléchargement ou paiement
        ↓
Commande envoyée à l’imprimeur
```

La prévisualisation et la génération finale doivent partager le même algorithme et les mêmes paramètres afin d’éviter une différence entre le résultat vu et le résultat acheté.

## 8. Modèle commercial

### Offres possibles

- aperçu gratuit avec filigrane ;
- PDF HD à l’unité ;
- impression A4 ;
- impression A3 ;
- packs ;
- abonnement pour utilisateurs réguliers ;
- offre B2B.

Hypothèses de prix à tester, sans validation de marché :

| Produit | Fourchette initiale |
|---|---:|
| PDF HD | 5 à 10 € |
| Tirage A4 | 15 à 25 € |
| Tirage A3 | 25 à 40 € |

Le prix final devra intégrer :

- impression ;
- emballage ;
- livraison ;
- frais de paiement ;
- infrastructure ;
- retours et support ;
- acquisition client ;
- marge.

### Impression à la demande ou imprimeur local

**Plateforme automatisée :**

- intégration rapide ;
- absence de stock ;
- couverture géographique large ;
- contrôle limité sur le papier et l’emballage.

**Imprimeur spécialisé local ou européen :**

- contrôle qualité supérieur ;
- meilleur choix de papiers ;
- marque blanche possible ;
- intégration et logistique plus manuelles.

La décision doit être prise après des essais physiques.

## 9. Sécurité, confidentialité et conformité

Les photos de personnes, particulièrement d’enfants, nécessitent une approche de protection des données dès la conception.

Mesures minimales :

- politique de confidentialité compréhensible ;
- chiffrement en transit et au repos ;
- fichiers privés accessibles par liens temporaires ;
- suppression automatique des originaux ;
- suppression manuelle des projets ;
- durées de conservation documentées ;
- contrôle des accès internes ;
- contrats avec les sous-traitants ;
- hébergement adapté aux utilisateurs européens ;
- procédure de violation de données ;
- absence d’entraînement de modèles sur les photos sans consentement spécifique.

Politique initiale envisageable :

- visiteurs : suppression des originaux sous 24 à 48 heures ;
- comptes : durée configurable et suppression à la demande ;
- commandes : conservation limitée au traitement et au support de la commande.

Il faut aussi prévoir des conditions d’utilisation et une gestion des contenus interdits avant toute transmission à un imprimeur.

## 10. Périmètre du MVP

### Inclus

- application web responsive ;
- import JPG et PNG ;
- recadrage A4 portrait et paysage ;
- trois niveaux de difficulté ;
- palette de 8 à 24 couleurs ;
- suppression ou fusion des micro-zones ;
- prévisualisation simplifiée et numérotée ;
- export SVG ;
- pack PDF avec dessin, modèle et palette ;
- compte simple ;
- paiement Stripe ;
- téléchargement sécurisé ;
- une option d’impression A4 ;
- suivi basique des commandes.

### Repoussé

- application desktop ;
- A3 et formats libres ;
- retouche manuelle zone par zone ;
- palette correspondant à une marque précise de crayons ;
- génération artistique par IA ;
- application mobile native ;
- livres complets ;
- marketplace ;
- impression encadrée ;
- internationalisation logistique avancée.

## 11. Principaux risques

| Risque | Réponse envisagée |
|---|---|
| Trop de petites zones | Seuil physique minimal, graphe de régions et fusions |
| Numéros illisibles | Transformée de distance, fusion et seuil de taille |
| Perte de ressemblance | Préservation des contours et pondération des zones importantes |
| Couleurs trop proches | Distance perceptuelle et avertissement |
| Preview différente du PDF | Pipeline commun et paramètres versionnés |
| Temps de calcul élevé | Preview réduite, cache et workers |
| Coût serveur | Recalcul partiel et génération HD après validation |
| Résultat correct à l’écran mais mauvais sur papier | Tests d’impression systématiques |
| Contenus inadaptés | Conditions, filtrage et procédure de modération |

## 12. POC technique à réaliser en premier

### Objectif

Valider la faisabilité et la qualité du moteur avant de construire les comptes, le paiement ou la commande d’impression.

Le POC est un outil local, sans authentification ni e-commerce, qui transforme une image en :

- image quantifiée ;
- carte des régions ;
- dessin avec contours ;
- dessin numéroté ;
- modèle coloré ;
- palette ;
- SVG ;
- PDF de test A4.

### Stack du POC

- Python 3.12 ;
- OpenCV ;
- NumPy ;
- scikit-image ;
- scikit-learn ;
- Pillow ;
- Shapely, si nécessaire pour les contours ;
- CairoSVG ou ReportLab pour le PDF ;
- interface légère avec Streamlit ou Gradio, après validation du script.

### Paramètres initiaux

- format : A4 portrait ;
- résolution de travail réglable ;
- couleurs : 8, 12, 16, 20 et 24 ;
- difficulté : simple, standard, détaillée ;
- nombre de superpixels ;
- surface minimale d’une région ;
- tolérance de fusion ;
- lissage des contours ;
- épaisseur de trait ;
- taille minimale des chiffres.

### Corpus de test

Commencer avec 20 à 30 images, puis atteindre 50 à 100 images :

- portraits seuls ;
- groupes ;
- animaux ;
- objets ;
- paysages ;
- arrière-plans simples ;
- arrière-plans complexes ;
- photos claires et sombres ;
- images nettes et légèrement floues ;
- sujets avec cheveux, pelage ou feuillage.

Les images doivent être utilisées avec les droits et consentements nécessaires.

### Métriques

Pour chaque génération, enregistrer :

- durée totale ;
- nombre de couleurs ;
- nombre de régions avant et après fusion ;
- surface de la plus petite région ;
- pourcentage de régions sous le seuil ;
- nombre de régions sans numéro lisible ;
- longueur et complexité des contours ;
- taille du SVG et du PDF.

Ajouter une évaluation humaine sur cinq critères, notés de 1 à 5 :

1. ressemblance avec la photo ;
2. propreté des contours ;
3. lisibilité des numéros ;
4. facilité à colorier ;
5. qualité de l’impression.

### Critères de succès proposés

Le POC peut être considéré comme suffisamment validé si, sur le corpus représentatif :

- au moins 80 % des images donnent un résultat utilisable sans retouche ;
- aucune région imprimée ne contient un numéro illisible dans le préréglage standard ;
- moins de 5 % des régions restantes se trouvent sous le seuil physique retenu ;
- la génération d’une preview prend moins de 5 secondes sur la machine de référence ;
- la génération finale prend moins de 30 secondes ;
- le SVG et le PDF restent visuellement cohérents ;
- les tests A4 sont réellement coloriables avec des crayons standards.

Ces seuils sont des objectifs de travail à ajuster après les premiers essais.

## 13. Plan d’exécution du POC

### Lot 1 — Baseline

- lecture et normalisation d’une image ;
- réduction de palette en CIELAB avec K-means ;
- création des composantes connexes ;
- export de l’image quantifiée et de statistiques.

**Livrable :** un script en ligne de commande et un dossier de résultats.

### Lot 2 — Régions coloriables

- SLIC ou autre segmentation ;
- graphe de voisinage ;
- fusion des petites zones ;
- lissage des contours ;
- comparaison de plusieurs stratégies.

**Livrable :** une carte de régions avec métriques avant/après.

### Lot 3 — Numérotation et SVG

- placement des numéros ;
- génération des contours ;
- création de la palette ;
- export SVG A4.

**Livrable :** un coloriage vectoriel imprimable.

### Lot 4 — PDF et tests papier

- PDF coloriage ;
- PDF modèle ;
- pack de trois pages ;
- impressions sur plusieurs papiers ;
- grille d’évaluation.

**Livrable :** un pack PDF et un rapport de test.

### Lot 5 — Interface de démonstration

- upload ;
- choix du nombre de couleurs ;
- sélection de la difficulté ;
- comparaison avant/après ;
- téléchargement.

**Livrable :** une petite application locale pour les tests utilisateurs.

## 14. Ordre de développement recommandé

1. Valider la quantification et la segmentation sur quelques images.
2. Résoudre les micro-zones et le placement des numéros.
3. Générer un SVG et un PDF A4.
4. Imprimer et colorier plusieurs résultats.
5. Tester le moteur sur un corpus plus large.
6. Ajouter l’interface de prévisualisation.
7. Faire tester le prototype par 10 à 20 personnes.
8. Construire ensuite les comptes, le paiement et l’impression à la demande.

## 15. Équipe et estimations indicatives

Compétences principales :

- développement frontend ;
- développement backend et traitement d’image ;
- design produit ;
- prépresse et tests papier ;
- sécurité et RGPD ;
- intégration paiement et impression.

Ordres de grandeur :

| Niveau | Délai indicatif | Budget indicatif |
|---|---:|---:|
| Prototype technique | 3 à 6 semaines | 8 000 à 20 000 € |
| MVP commercial | 2 à 4 mois | 30 000 à 70 000 € |
| Version aboutie | 6 à 12 mois | 80 000 à 200 000 € et plus |

La principale variable est le niveau de qualité attendu sur des photos très différentes. Une démonstration est rapide à produire ; un moteur robuste sur les portraits, animaux, paysages et arrière-plans complexes demande beaucoup de tests.

## 16. Décision immédiate

La prochaine étape est de construire le **POC du moteur**, sans interface commerciale.

La première itération doit répondre à quatre questions :

1. La réduction de palette conserve-t-elle suffisamment la ressemblance ?
2. Peut-on éliminer automatiquement les micro-zones ?
3. Les numéros restent-ils lisibles après impression A4 ?
4. Quels paramètres donnent un bon compromis entre fidélité et facilité ?

Une fois ces réponses obtenues sur papier, le projet pourra avancer vers une interface web, puis vers le MVP commercial.
