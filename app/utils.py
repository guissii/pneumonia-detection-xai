"""
utils.py
────────
Fonctions utilitaires pour l'application Streamlit :
  - load_model      : charge ResNet-18 depuis le fichier .pt
  - predict         : lance l'inférence sur une image PIL
  - compute_gradcam : calcule la heatmap Grad-CAM superposée

Ces fonctions sont indépendantes de l'interface et peuvent être
testées seules (cf. tests/test_app_logic.py).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import resnet18

# ── Constantes ────────────────────────────────────────────────────────────────

CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Chargement du modèle ──────────────────────────────────────────────────────

def load_model(model_path: str) -> nn.Module:
    """
    Charge ResNet-18 depuis un fichier .pt et retourne le modèle en mode eval.

    Args:
        model_path: chemin vers best_resnet18.pt

    Returns:
        nn.Module en mode eval, sur CPU ou GPU selon disponibilité

    Raises:
        FileNotFoundError: si le fichier .pt est introuvable
        RuntimeError: si le chargement PyTorch échoue (fichier corrompu, etc.)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Assurez-vous que notebooks/best_resnet18.pt est présent."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model = model.to(device).eval()
    return model


# ── Prédiction ────────────────────────────────────────────────────────────────

def predict(model: nn.Module, image: Image.Image) -> dict:
    """
    Lance l'inférence sur une image PIL et retourne un dictionnaire de résultats.

    Args:
        model : modèle chargé via load_model()
        image : PIL.Image (mode quelconque, converti en RGB en interne)

    Returns:
        {
            "predicted_class": str,               # "NORMAL" ou "PNEUMONIA"
            "confidence":      float,             # prob. de la classe prédite [0, 1]
            "probabilities":   dict[str, float],  # prob. pour chaque classe
            "tensor":          torch.Tensor,      # [1, 3, 224, 224] pour Grad-CAM
            "pred_idx":        int,               # 0 = NORMAL, 1 = PNEUMONIA
        }

    Raises:
        ValueError: si image n'est pas une PIL.Image valide
    """
    if not isinstance(image, Image.Image):
        raise ValueError("L'argument 'image' doit être un objet PIL.Image.")

    device = next(model.parameters()).device

    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu()

    pred_idx   = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    prob_dict  = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence":      confidence,
        "probabilities":   prob_dict,
        "tensor":          tensor,
        "pred_idx":        pred_idx,
    }


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Dé-normalise un tensor [3, H, W] → numpy [H, W, 3] float32 [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy().astype(np.float32)


def compute_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    pred_idx: int,
) -> Image.Image:
    """
    Calcule la heatmap Grad-CAM et retourne l'image superposée (PIL.Image).

    Args:
        model    : ResNet-18 en mode eval
        tensor   : tensor [1, 3, 224, 224] issu de predict()
        pred_idx : index de la classe prédite (0=NORMAL, 1=PNEUMONIA)

    Returns:
        PIL.Image RGB (224×224) — image originale + heatmap superposée
    """
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(pred_idx)],
    )
    heatmap = grayscale_cam[0]                     # [224, 224] float [0, 1]
    img_rgb = _denormalize(tensor.squeeze(0))       # [224, 224, 3]

    overlay = show_cam_on_image(img_rgb, heatmap, use_rgb=True, image_weight=0.5)
    return Image.fromarray(overlay)
