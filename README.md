# Pneumonia Detection & Explainability (XAI) with Grad-CAM

Classification automatique de radiographies thoraciques (**NORMAL / PNEUMONIA**) assistee par ResNet-18 et cartes d'attention **Grad-CAM**.

> ⚠️ **Avertissement médical** — Projet à vocation strictement pédagogique et de recherche. Cet outil **ne constitue pas un dispositif médical** et ne remplace en aucun cas le jugement clinique d'un professionnel de santé.

---

## 🎯 Objectif du projet

Ce projet propose une solution complète de vision par ordinateur pour le dépistage assisté de la pneumonie à partir de radiographies pulmonaires :
1. **Classification robuste** : Modèle ResNet-18 adapté en Transfer Learning sur le benchmark Kaggle Kermany et al.
2. **Explicabilité médicale (XAI)** : Implémentation de **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) pour matérialiser les zones pulmonaires ayant motivé le résultat.
3. **Application interactive prête à l'emploi** : Interface Streamlit moderne, responsive et conteneurisée avec Docker.

---

## 📊 Résultats et Performances du Modèle

Évalué sur le jeu de test indépendant (**624 radiographies**, Kermany et al.) :

| Métrique | Valeur obtenue | Interprétation clinique |
| :--- | :---: | :--- |
| **Recall (Sensibilité) Pneumonie** | **98.2 %** | **Métrique critique** : seulement 7 faux négatifs sur 390 cas pathologiques. |
| **Précision Pneumonie** | **81.3 %** | Minimise les doutes sur les alertes générées. |
| **F1-Score Pneumonie** | **89.0 %** | Excellent équilibre entre détection et précision. |
| **Accuracy Globale** | **84.8 %** | Bonne séparation globale sur le jeu de test complet. |
| **Spécificité (NORMAL)** | **62.4 %** | Biais conservateur favorisant la sécurité du patient (sur-détection vs omission). |

### Matrice de Confusion
![Matrice de Confusion](images/confusion_matrix.png)

---

## 🧠 Explicabilité avec Grad-CAM

La méthode Grad-CAM projette les gradients de la dernière couche convolutive (`layer4` de ResNet-18) pour produire une carte de chaleur superposée :
- **Rouge / Jaune (activation forte)** : Zones déterminantes pour le diagnostic (opacités alvéolaires, foyers de condensation).
- **Bleu / Vert (faible influence)** : Régions neutres ou saines.

![Comparaison Grad-CAM](images/gradcam_comparison.png)

---

## 🚀 Démarrage Rapide

### Option A — Avec Docker (Recommandé, zéro configuration)

```bash
# Cloner le projet
git clone https://github.com/guissii/pneumonia-detection-xai.git
cd pneumonia-detection-xai

# Lancer l'application conteneurisée
docker-compose up --build
```
L'application est immédiatement accessible sur [http://localhost:8501](http://localhost:8501).

---

### Option B — Installation locale avec Python

**Prérequis** : Python 3.10+ et le fichier de poids `notebooks/best_resnet18.pt`.

```bash
# 1. Cloner et se placer dans le projet
git clone https://github.com/guissii/pneumonia-detection-xai.git
cd pneumonia-detection-xai

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application Streamlit
streamlit run app/app.py
```

---

## 🧪 Tests Unitaires

Le projet inclut une suite de tests automatisés pour valider la logique d'inférence, la robustesse aux entrées invalides et le calcul Grad-CAM :

```bash
pytest tests/test_app_logic.py -v
```

---

## 📂 Architecture du Répertoire

```
pneumonia-detection-xai/
├── notebooks/
│   ├── 01_cnn_principe_et_classification.ipynb  # Exploration CNN & entraînement ResNet-18
│   ├── 02_gradcam_explicabilite.ipynb           # Analyse fine et validation Grad-CAM
│   └── best_resnet18.pt                         # Poids du modèle entraîné
├── app/
│   ├── app.py       # Interface utilisateur Streamlit
│   └── utils.py     # Logique de chargement, prédiction et génération Grad-CAM
├── tests/
│   └── test_app_logic.py  # Tests unitaires indépendants de l'UI
├── images/          # Graphiques, matrices et comparatifs visuels
├── Dockerfile       # Configuration de l'image de production
├── docker-compose.yml
├── requirements.txt # Dépendances minimales optimisées
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 💡 Note de conception : Pourquoi ce projet reste pragmatique

Contrairement à des architectures complexes de streaming ou de MLOps lourd (Kubernetes, pipelines de réentraînement continu) :
- Les critères radiologiques de la pneumonie sont **stables dans le temps** (pas de concept drift rapide).
- L'inférence locale et conteneurisée garantit **simplicité de déploiement, faible latence et confidentialité des données**.
- Dans un cadre de production hospitalière réelle, les priorités se concentreraient sur le **marquage réglementaire (CE/FDA)**, l'**intégration PACS/DICOM** et la **validation clinique multicentrique**.

---

## 📚 Référence Dataset

> Kermany, Daniel; Goldbaum, Michael; Cai, Wentao et al. (2018), *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, Cell, 172(5), 1122-1131. [Lien Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
