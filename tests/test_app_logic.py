"""
test_app_logic.py
─────────────────
Tests des fonctions de utils.py, indépendants de l'interface Streamlit.

Lancement :
    pytest tests/test_app_logic.py -v

Pré-requis :
    - notebooks/best_resnet18.pt doit être présent (tests de chargement/prédiction)
    - pip install -r requirements.txt
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ── Chemins ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH   = PROJECT_ROOT / "notebooks" / "best_resnet18.pt"
APP_DIR      = PROJECT_ROOT / "app"

# utils.py est dans app/ — on l'ajoute au sys.path pour l'import
import sys
sys.path.insert(0, str(APP_DIR))

from utils import CLASS_NAMES, compute_gradcam, load_model, predict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model():
    """Charge le modèle une seule fois pour toute la session de tests."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Modèle absent : {MODEL_PATH} — skip des tests d'inférence.")
    return load_model(MODEL_PATH)


@pytest.fixture
def dummy_rgb_image():
    """Image RGB 224×224 avec pixels aléatoires."""
    rng  = np.random.default_rng(42)
    data = (rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    return Image.fromarray(data, mode="RGB")


@pytest.fixture
def dummy_grayscale_image():
    """Image en niveaux de gris (comme certaines radios)."""
    rng  = np.random.default_rng(0)
    data = (rng.integers(0, 256, (224, 224), dtype=np.uint8))
    return Image.fromarray(data, mode="L")


# ── Tests : chargement du modèle ──────────────────────────────────────────────

class TestLoadModel:

    def test_load_model_returns_nn_module(self, model):
        """Le modèle chargé doit être un nn.Module."""
        import torch.nn as nn
        assert isinstance(model, nn.Module)

    def test_model_in_eval_mode(self, model):
        """Le modèle doit être en mode eval (pas d'entraînement)."""
        assert not model.training

    def test_model_has_2_output_classes(self, model):
        """La couche fc finale doit avoir 2 sorties (NORMAL/PNEUMONIA)."""
        assert model.fc.out_features == 2

    def test_load_model_raises_on_missing_file(self):
        """FileNotFoundError si le .pt n'existe pas."""
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_model("/chemin/inexistant/modele.pt")


# ── Tests : prédiction ────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_returns_expected_keys(self, model, dummy_rgb_image):
        """Le dictionnaire retourné doit contenir toutes les clés attendues."""
        result = predict(model, dummy_rgb_image)
        expected_keys = {"predicted_class", "confidence", "probabilities", "tensor", "pred_idx"}
        assert expected_keys == set(result.keys())

    def test_predict_class_is_valid(self, model, dummy_rgb_image):
        """La classe prédite doit être NORMAL ou PNEUMONIA."""
        result = predict(model, dummy_rgb_image)
        assert result["predicted_class"] in CLASS_NAMES

    def test_predict_confidence_in_range(self, model, dummy_rgb_image):
        """La confiance doit être un float entre 0 et 1."""
        result = predict(model, dummy_rgb_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, model, dummy_rgb_image):
        """Les probabilités doivent sommer à 1 (à une tolérance numérique près)."""
        result = predict(model, dummy_rgb_image)
        total  = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_predict_tensor_shape(self, model, dummy_rgb_image):
        """Le tensor retourné doit avoir la forme [1, 3, 224, 224]."""
        result = predict(model, dummy_rgb_image)
        assert result["tensor"].shape == (1, 3, 224, 224)

    def test_predict_pred_idx_matches_class(self, model, dummy_rgb_image):
        """pred_idx doit correspondre à la classe prédite dans CLASS_NAMES."""
        result = predict(model, dummy_rgb_image)
        assert CLASS_NAMES[result["pred_idx"]] == result["predicted_class"]

    def test_predict_accepts_grayscale_image(self, model, dummy_grayscale_image):
        """Les images en niveaux de gris doivent être acceptées (conversion RGB)."""
        result = predict(model, dummy_grayscale_image)
        assert result["predicted_class"] in CLASS_NAMES

    def test_predict_raises_on_invalid_input(self, model):
        """ValueError si l'entrée n'est pas une PIL.Image."""
        with pytest.raises(ValueError, match="PIL.Image"):
            predict(model, "pas_une_image")

    def test_predict_raises_on_numpy_array(self, model):
        """ValueError si l'entrée est un numpy array (pas une PIL.Image)."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            predict(model, arr)


# ── Tests : Grad-CAM ──────────────────────────────────────────────────────────

class TestGradCAM:

    def test_gradcam_returns_pil_image(self, model, dummy_rgb_image):
        """compute_gradcam doit retourner une PIL.Image."""
        result     = predict(model, dummy_rgb_image)
        cam_image  = compute_gradcam(model, result["tensor"], result["pred_idx"])
        assert isinstance(cam_image, Image.Image)

    def test_gradcam_output_is_rgb(self, model, dummy_rgb_image):
        """L'image Grad-CAM doit être en mode RGB."""
        result    = predict(model, dummy_rgb_image)
        cam_image = compute_gradcam(model, result["tensor"], result["pred_idx"])
        assert cam_image.mode == "RGB"

    def test_gradcam_output_size(self, model, dummy_rgb_image):
        """L'image Grad-CAM doit être 224×224."""
        result    = predict(model, dummy_rgb_image)
        cam_image = compute_gradcam(model, result["tensor"], result["pred_idx"])
        assert cam_image.size == (224, 224)

    def test_gradcam_works_for_both_classes(self, model):
        """Grad-CAM doit fonctionner pour pred_idx=0 et pred_idx=1."""
        rng   = np.random.default_rng(7)
        data  = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(data, mode="RGB")
        result = predict(model, image)

        tensor = result["tensor"]
        for pred_idx in [0, 1]:
            cam = compute_gradcam(model, tensor, pred_idx)
            assert isinstance(cam, Image.Image)
