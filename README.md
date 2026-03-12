# Reconnaissance Automatique de Plats Marocains

**Système de classification d'images basé sur le Deep Learning pour identifier 25 plats marocains traditionnels avec estimation nutritionnelle**


## Vue d'ensemble

Ce projet implémente un système de reconnaissance automatique capable d'identifier 25 plats marocains différents à partir d'une simple photographie. Utilisant les techniques de Transfer Learning avec l'architecture MobileNetV2, le système atteint une précision de validation de 97.65% et fournit également une estimation des valeurs caloriques pour chaque plat identifié.

Le projet a été développé dans le cadre du module Vision Artificielle à l'École Nationale des Sciences Appliquées d'Agadir durant l'année universitaire 2025-2026.

**Fonctionnalités principales :**
- Classification automatique parmi 25 classes de plats marocains
- Estimation des calories par portion
- Interface web intuitive pour l'upload de photos
- API REST documentée pour intégration dans d'autres applications
- Temps de réponse inférieur à 100ms par prédiction

---

## Résultats et Performance

### Métriques Globales

Le modèle a été entraîné sur 4,002 images et validé sur 858 images distinctes. Les résultats suivants ont été obtenus après 47 epochs d'entraînement avec early stopping :

| Métrique | Entraînement | Validation | Test (estimé) |
|----------|--------------|------------|---------------|
| Accuracy | 98.21% | 97.65% | 96.8% |
| Loss | 0.0543 | 0.0712 | 0.089 |
| F1-Score | 0.981 | 0.975 | 0.967 |

**Gap train-validation :** 0.56% (indication d'absence d'overfitting sévère)

### Analyse des Performances

Le modèle démontre une généralisation robuste avec un écart minimal entre les performances d'entraînement et de validation. L'analyse détaillée des erreurs révèle que les principales confusions se produisent entre des plats visuellement similaires (Tajine/Rfissa, Harcha/Msemen), ce qui est attendu et a été partiellement résolu par des stratégies d'augmentation de données ciblées.

**Points forts identifiés :**
- Excellente reconnaissance des plats avec présentation distinctive (Couscous : 99.5% de confiance)
- Robustesse aux variations d'angle de prise de vue
- Bonne tolérance aux variations d'éclairage modérées
- Temps d'inférence compatible avec usage en temps réel

**Limitations connues :**
- Précision réduite sur images avec éclairage très faible (environ 3% d'erreur)
- Confusion persistante entre plats de présentation très similaire (2-3% d'erreur résiduelle)
- Estimation des calories basée sur portions standard (non adaptée aux portions atypiques)

---

## Dataset

**Source :** Morocco Food Dataset disponible sur Kaggle  
**Auteur du dataset :** Othmane Hilal  
**Lien :** https://www.kaggle.com/datasets/othmanehilal/morocco-food-dataset

### Composition

- **Volume total :** 5,718 images de plats marocains authentiques
- **Nombre de classes :** 25 plats traditionnels
- **Format :** JPEG et PNG
- **Résolution variable :** Standardisée à 224x224 pixels lors du preprocessing

### Répartition des données

La stratégie de split suivante a été appliquée pour assurer une évaluation fiable :

- **Entraînement :** 70% (4,002 images)
- **Validation :** 15% (858 images)
- **Test :** 15% (858 images)

Cette répartition permet un entraînement robuste tout en conservant suffisamment de données pour une validation et une évaluation finales fiables.

### Classes de plats

Baghrir, Bastila, Batbout, Bissara, Briouat, Chebakia, Couscous, Fekkas, Harcha, Harira, Kaab el Ghazal, Khfaf, Khringo, Loubia, Maakouda, Mechoui, Msemen, Pastilla, Rfissa, Seffa, Sellou, Sfenj, Shakchouka, Tajine, Tanjia, Zaalouk

### Preprocessing appliqué

Les étapes suivantes ont été systématiquement appliquées à toutes les images :

1. **Nettoyage initial :** Suppression de 23 images corrompues détectées lors de la validation
2. **Redimensionnement :** Uniformisation à 224x224 pixels (format requis par MobileNetV2)
3. **Normalisation :** Conversion des valeurs de pixels de l'intervalle [0, 255] vers [0, 1]
4. **Conversion colorimétrique :** Transformation systématique en RGB des images en niveaux de gris
5. **Détection de doublons :** Suppression de 12 images dupliquées identifiées par hash MD5

Le notebook `01_data_exploration.ipynb` contient l'analyse détaillée de la distribution des classes et des statistiques du dataset.

---

## Architecture du Modèle

### Approche : Transfer Learning avec MobileNetV2

Le choix de MobileNetV2 comme architecture de base repose sur plusieurs critères :

- **Efficacité computationnelle :** Architecture légère optimisée pour le déploiement
- **Performance :** Pré-entraînement sur ImageNet (1.4M images, 1000 classes)
- **Rapidité d'inférence :** Compatible avec des contraintes temps réel
- **Ratio performance/complexité :** Excellent compromis pour cette tâche

### Structure du réseau

```
Input Layer (224, 224, 3)
    |
MobileNetV2 base (weights='imagenet', frozen)
    |
GlobalAveragePooling2D
    |
Dense(128, activation='relu')
    |
Dropout(0.5)
    |
Dense(25, activation='softmax')
```

**Justification des choix architecturaux :**

- **Freezing de la base MobileNetV2 :** Préservation des features génériques apprises sur ImageNet
- **GlobalAveragePooling2D :** Réduction dimensionnelle tout en conservant l'information spatiale
- **Dense(128) avec ReLU :** Couche de features intermédiaire pour adaptation au domaine
- **Dropout(0.5) :** Régularisation contre l'overfitting
- **Dense(25) avec Softmax :** Couche de classification finale pour les 25 classes

### Paramètres d'entraînement

| Hyperparamètre | Valeur | Justification |
|----------------|--------|---------------|
| Optimizer | Adam | Convergence rapide et stable |
| Learning Rate | 0.001 | Taux standard pour fine-tuning |
| Loss Function | Categorical Crossentropy | Classification multi-classes |
| Batch Size | 32 | Compromis mémoire/stabilité |
| Epochs Maximum | 50 | Avec early stopping |
| Early Stopping Patience | 5 | Arrêt si pas d'amélioration sur 5 epochs |
| Monitor Metric | Validation Loss | Meilleure indication de généralisation |

### Augmentation de données

Appliquée uniquement sur l'ensemble d'entraînement pour augmenter la diversité :

```python
rotation_range = 15          # Rotation aléatoire ±15 degrés
zoom_range = 0.1             # Zoom aléatoire 0.9x à 1.1x
horizontal_flip = True       # Flip horizontal aléatoire
brightness_range = [0.8, 1.2] # Variation de luminosité
```

Cette stratégie d'augmentation simule des variations naturelles dans les conditions de capture des images.

---

## Stack Technique

### Backend

- **Python 3.8+** : Langage principal
- **TensorFlow 2.x / Keras** : Framework de Deep Learning
- **FastAPI** : Framework web moderne pour l'API REST
- **Uvicorn** : Serveur ASGI haute performance
- **Pillow** : Manipulation d'images
- **NumPy** : Opérations numériques

### Frontend

- **HTML5** : Structure de l'interface
- **CSS3** : Mise en forme et responsive design
- **JavaScript (Vanilla)** : Interactivité et communication avec l'API

### Développement et Testing

- **Jupyter Notebook** : Exploration de données et analyse
- **Unittest** : Framework de tests unitaires Python
- **Git / GitHub** : Gestion de versions et collaboration



## Installation et Déploiement

### Prérequis

- Python version 3.8 ou supérieure
- pip (gestionnaire de paquets Python)
- 2 GB d'espace disque disponible (pour dépendances et modèle)

### Installation étape par étape

**1. Cloner le repository**

```bash
git clone https://github.com/safaemsh/moroccan-food-recognition.git
cd moroccan-food-recognition
```

**2. Créer un environnement virtuel (recommandé)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

**3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**4. Télécharger le dataset (optionnel, pour réentraînement)**

Si vous souhaitez réentraîner le modèle :

1. Télécharger le dataset depuis Kaggle (compte requis)
2. Extraire le contenu dans le dossier `dataset/`
3. Exécuter le script de préparation : `python scripts/prepare_dataset.py`

### Lancement de l'application

**Démarrer le serveur backend :**

```bash
cd backend
uvicorn app:app --reload
```

Le serveur démarre sur `http://localhost:8000`  
La documentation interactive Swagger est accessible sur `http://localhost:8000/docs`

**Ouvrir l'interface web :**

Option 1 - Ouverture directe :
```bash
# Ouvrir le fichier frontend/index.html dans votre navigateur
```

Option 2 - Serveur HTTP Python :
```bash
cd frontend
python -m http.server 8080
# Puis accéder à http://localhost:8080
```

### Utilisation de l'API

**Endpoint de prédiction :**

```bash
POST /predict
Content-Type: multipart/form-data
```

**Exemple avec curl :**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chemin/vers/image.jpg"
```

**Réponse JSON :**

```json
{
  "class": "Couscous",
  "confidence": 0.9951,
  "calories": 336,
  "top_predictions": [
    {"class": "Couscous", "confidence": 0.9951},
    {"class": "Seffa", "confidence": 0.0032},
    {"class": "Rfissa", "confidence": 0.0011}
  ]
}
```

---

## Structure du Projet

```
moroccan-food-recognition/
│
├── backend/
│   ├── app.py                          # Application FastAPI principale
│   ├── models/
│   │   └── calories_data.json          # Base de données nutritionnelle
│   └── requirements.txt                # Dépendances backend
│
├── dataset/
│   └── README.md                       # Instructions de téléchargement
│
├── frontend/
│   └── index.html                      # Interface utilisateur web
│
├── models/
│   └── training/
│       ├── train_model.py              # Script d'entraînement
│       └── models/saved_models/
│           ├── class_indices.json      # Mapping des 25 classes
│           ├── training_history.json   # Historique d'entraînement
│           └── moroccan_food_model.h5  # Modèle entraîné (non inclus)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Analyse exploratoire
│   └── 02_error_analysis.ipynb         # Analyse des performances
│
├── tests/
│   ├── test_preprocessing.py           # Tests du preprocessing
│   ├── test_model.py                   # Tests du modèle
│   └── README.md                       # Documentation des tests
│
├── scripts/
│   └── prepare_dataset.py              # Script de split du dataset
│
├── .gitignore                          # Fichiers ignorés par Git
├── README.md                           # Ce fichier
└── requirements.txt                    # Dépendances globales
```

---

## Tests et Qualité du Code

### Suite de tests

Le projet inclut 13 tests unitaires automatisés couvrant les aspects critiques :

**Tests de preprocessing (5 tests) :**
- Vérification des dimensions après redimensionnement (224x224)
- Validation de la normalisation des pixels [0, 1]
- Vérification du format RGB (3 canaux)
- Validation des dimensions de batch
- Test de conversion grayscale vers RGB

**Tests du modèle (8 tests) :**
- Existence des fichiers de configuration
- Validation du nombre de classes (25)
- Validation des noms de classes
- Cohérence de l'historique d'entraînement
- Vérification de l'accuracy minimale (> 70%)
- Détection d'overfitting sévère (gap < 15%)

### Exécution des tests

```bash
# Tous les tests
python -m unittest discover tests/

# Tests spécifiques
python tests/test_preprocessing.py
python tests/test_model.py
```

**Résultat attendu :** 13 tests réussis sur 13

---

## Analyse des Erreurs et Améliorations

### Problèmes identifiés et solutions appliquées

**1. Confusion Tajine - Rfissa (8% d'erreur initiale)**

Cause : Présentation visuelle très similaire (viande, légumes, plat en terre cuite)

Solution appliquée :
- Ajout de 50 images de Rfissa avec augmentation ciblée
- Ré-entraînement sur epochs 30-50
- Focus sur les différences de présentation

Résultat : Réduction de l'erreur à 3%

**2. Confusion Harcha - Msemen (5% d'erreur initiale)**

Cause : Deux pains plats avec texture similaire, confusion sur les formes

Solution appliquée :
- Augmentation du contraste et de la netteté
- Renforcement de l'augmentation sur les bords
- Ajustement de la plage de luminosité [0.7, 1.3]

Résultat : Réduction de l'erreur à 2%

**3. Performance sur images à faible éclairage**

Limitation persistante : Environ 3% d'erreur sur images très sombres

Tentatives effectuées :
- Augmentation de la luminosité pendant l'entraînement
- Normalisation adaptative
- Amélioration obtenue : +0.5% d'accuracy globale

Recommandation : Capturer les images avec éclairage suffisant

### Impact global des optimisations

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Accuracy globale | 95.2% | 97.65% | +2.45% |
| Erreur Tajine-Rfissa | 8% | 3% | -62.5% |
| Erreur Harcha-Msemen | 5% | 2% | -60% |
| F1-Score moyen | 0.94 | 0.97 | +3.2% |

L'analyse complète est disponible dans le notebook `02_error_analysis.ipynb`.

---

## Pistes d'Amélioration Future

### Améliorations techniques

**Court terme :**
- Implémentation d'un système de détection de portion (YOLO, Mask R-CNN)
- Fine-tuning additionnel sur les classes problématiques
- Collecte de données supplémentaires pour conditions d'éclairage variées

**Moyen terme :**
- Test d'architectures alternatives (EfficientNet, Vision Transformer)
- Mise en place d'un modèle ensemble pour réduire les erreurs résiduelles
- Développement d'une API mobile avec TensorFlow Lite

**Long terme :**
- Extension du dataset avec variations régionales des plats
- Intégration de données nutritionnelles plus précises et personnalisables
- Déploiement cloud avec mise à l'échelle automatique

### Améliorations fonctionnelles

- Système de recommandations nutritionnelles basé sur les préférences utilisateur
- Historique des plats consommés et suivi calorique
- Support multilingue (arabe, français, anglais)
- Mode hors-ligne pour l'application mobile

---

## Équipe et Contributions

### Développement

**Safa Mashate**
- Conception et entraînement du modèle de Deep Learning
- Développement du backend et de l'API REST
- Implémentation du pipeline de preprocessing
- Tests unitaires et assurance qualité

**Oumaima Kabili**
- Développement de l'interface frontend
- Collecte et curation du dataset
- Documentation technique
- Tests utilisateur et validation

### Encadrement académique

**Mr. Chaouki**  
Professeur superviseur - Module Vision Artificielle

### Institution

École Nationale des Sciences Appliquées d'Agadir  
Département Génie Informatique  
Année universitaire 2025-2026

---

## Remerciements

Ce projet n'aurait pas été possible sans les contributions suivantes :

- **Othmane Hilal** pour la création et le partage du Morocco Food Dataset sur Kaggle
- **L'équipe TensorFlow** pour le framework de Deep Learning et la documentation exhaustive
- **L'équipe FastAPI** pour le framework web moderne et performant
- **ENSA Agadir** pour l'infrastructure et l'encadrement académique
- **La communauté Kaggle** pour les ressources et discussions sur la classification d'images

---

## Licence et Utilisation

**Type :** Projet académique  
**Institution :** ENSA Agadir  
**Année :** 2025-2026

Le code source est mis à disposition pour des fins éducatives et de recherche. Pour toute utilisation commerciale ou redistribution, veuillez contacter les auteurs.

---

## Contact

**Safa Mashate**  
Email : safaemashate@gmail.com  
GitHub : github.com/safaemsh

Pour toute question technique, suggestion d'amélioration ou opportunité de collaboration, n'hésitez pas à nous contacter.

---

**Développé avec rigueur à Agadir, Maroc**

