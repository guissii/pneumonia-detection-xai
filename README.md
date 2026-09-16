# Chest X-Ray Diagnostic Assistant

Classification automatique de radiographies thoraciques (**NORMAL / PNEUMONIA**)
avec explicabilitÃ© Grad-CAM.

> âš ï¸ **Avertissement mÃ©dical** â€” Ceci est un outil pÃ©dagogique, pas un dispositif
> mÃ©dical. Les rÃ©sultats affichÃ©s ne remplacent pas un avis mÃ©dical professionnel.
> Ne prenez aucune dÃ©cision clinique sur la base de cet outil.

---

## Objectif

Ce projet illustre comment un rÃ©seau de neurones convolutif (ResNet-18 en
transfer learning) peut classer des radiographies thoraciques, et comment
l'explicabilitÃ© Grad-CAM permet de visualiser les zones d'attention du modÃ¨le.

Il est volontairement simple : une interface graphique locale, un modÃ¨le
prÃ©-entraÃ®nÃ©, des tests unitaires. Pas de pipeline MLOps â€” voir
[Pourquoi ce projet reste simple](#pourquoi-ce-projet-reste-simple).

---

## Structure du projet

```
chest-xray-diagnostic-assistant/
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_cnn_principe_et_classification.ipynb  â€” Cours CNN + entraÃ®nement ResNet-18
â”‚   â”œâ”€â”€ 02_gradcam_explicabilite.ipynb           â€” Grad-CAM interactif
â”‚   â””â”€â”€ best_resnet18.pt                         â€” ModÃ¨le entraÃ®nÃ© (non versionnÃ©)
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ app.py       â€” Interface Streamlit
â”‚   â””â”€â”€ utils.py     â€” Chargement modÃ¨le, prÃ©diction, Grad-CAM
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_app_logic.py  â€” Tests unitaires de utils.py
â”œâ”€â”€ images/          â€” Outputs (matrice de confusion, etc.)
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â””â”€â”€ README.md
```

---

## RÃ©sultats du modÃ¨le

Ã‰valuÃ©s sur le test set du dataset Chest X-Ray Kaggle
(Kermany et al., 2018 â€” 624 images).

| MÃ©trique             | Valeur |
|----------------------|--------|
| Accuracy             | > 95 % |
| Recall PNEUMONIA     | > 95 % |
| SpÃ©cificitÃ© (NORMAL) | > 90 % |
| F1-Score PNEUMONIA   | > 95 % |

Le recall PNEUMONIA est la mÃ©trique prioritaire : un faux nÃ©gatif (pneumonie
non dÃ©tectÃ©e) est cliniquement plus grave qu'un faux positif.

---

## Installation et lancement

### PrÃ©requis

- Python 3.10+ recommandÃ©
- `notebooks/best_resnet18.pt` prÃ©sent (gÃ©nÃ©rÃ© par le notebook 01, non versionnÃ©)

### Installation

```bash
pip install -r requirements.txt
```

> **Note PyTorch** : pour installer une version avec support GPU (CUDA), remplacez
> la ligne `torch` dans `requirements.txt` par :
> ```
> pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
> ```

### Lancer l'application

```bash
streamlit run app/app.py
```

L'interface s'ouvre automatiquement dans le navigateur (http://localhost:8501).

### Utilisation

1. Uploadez une radiographie thoracique (JPEG ou PNG)
2. Cliquez sur **Lancer l'analyse**
3. L'interface affiche :
   - La classe prÃ©dite (**NORMAL** ou **PNEUMONIA**) avec le score de confiance
   - Une barre de progression visuelle du score
   - L'image originale et la heatmap Grad-CAM cÃ´te Ã  cÃ´te

---

## Tests

```bash
# Tous les tests (nÃ©cessite best_resnet18.pt)
pytest tests/test_app_logic.py -v

# Test unique : validation des entrÃ©es invalides (sans modÃ¨le)
pytest tests/test_app_logic.py::TestLoadModel::test_load_model_raises_on_missing_file -v
```

Les tests couvrent :
- Chargement du modÃ¨le (prÃ©sence du fichier, mode eval, architecture)
- Format de sortie de `predict()` (clÃ©s, types, contraintes numÃ©riques)
- Gestion d'une entrÃ©e invalide (pas de PIL.Image)
- Sortie de `compute_gradcam()` (type PIL.Image, mode RGB, taille 224Ã—224)

---

## Pourquoi ce projet reste volontairement simple

La nature du problÃ¨me ne justifie pas un pipeline MLOps complet :

**Classification d'images mÃ©dicales stable dans le temps.**
Les radiographies thoraciques (NORMAL / PNEUMONIA bactÃ©rienne ou virale) suivent
des critÃ¨res radiologiques stables. Il n'y a pas de dÃ©rive conceptuelle liÃ©e au
comportement utilisateur, Ã  la saisonnalitÃ©, ou Ã  l'Ã©volution rapide du domaine.
Le modÃ¨le n'a pas besoin d'Ãªtre rÃ©entraÃ®nÃ© en continu.

**VolumÃ©trie modeste et latence non critique.**
Ce projet n'est pas un service en production recevant des milliers de requÃªtes
par seconde. L'interface locale Streamlit rÃ©pond en < 2 s pour une image, sans
besoin de scaling horizontal ni d'orchestration Kubernetes.

**Overhead disproportionnÃ©.**
Maintenir Docker, Kubernetes, GitHub Actions CI/CD, MLflow et Prometheus/Grafana
pour un modÃ¨le de classification binaire sur un dataset figÃ© reprÃ©sente un coÃ»t
de maintenance qui n'apporte aucune valeur ajoutÃ©e Ã  ce cas d'usage.

**Ce qui serait justifiÃ© dans un contexte de production clinique rÃ©elle :**
- Certification CE (EU AI Act, classe haut risque) avant tout usage clinique
- Anonymisation RGPD des images radiographiques avant stockage
- TraÃ§abilitÃ© de chaque prÃ©diction (qui, quand, quel modÃ¨le, quel rÃ©sultat)
- API sÃ©curisÃ©e avec authentification (pas une interface Streamlit publique)
- Monitoring de la qualitÃ© des images en entrÃ©e (pas de drift de performance)

---

## Dataset

Kermany, D., Goldbaum, M., Cai, W., et al. (2018). *Identifying Medical Diagnoses
and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122â€“1131.

Dataset disponible sur Kaggle :
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Le dataset (5,8 Go) n'est pas inclus dans ce dÃ©pÃ´t. Pour rÃ©entraÃ®ner, placer
le contenu dans `archive/chest_xray/` avec la structure `train/val/test / NORMAL/PNEUMONIA`.

