"""
app.py
──────
Interface Streamlit pour le Chest X-Ray Diagnostic Assistant.

Lancement :
    streamlit run app/app.py

Le modèle est chargé une seule fois au démarrage via st.cache_resource.
Toute la logique métier (modèle, prédiction, Grad-CAM) est dans utils.py.
"""

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

# Chemin vers le modèle — résolu par rapport à la racine du projet
_MODEL_PATH = Path(__file__).parent.parent / "notebooks" / "best_resnet18.pt"

# ── Import utils (avec message d'erreur si dépendances manquantes) ────────────
try:
    from utils import load_model, predict, compute_gradcam
except ImportError as e:
    st.error(
        f"❌ Impossible d'importer les dépendances : {e}\n\n"
        "Installez les paquets requis :\n```\npip install -r requirements.txt\n```"
    )
    st.stop()


# ── Configuration de la page ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Chest X-Ray Diagnostic Assistant",
    page_icon="🫁",
    layout="wide",
)


# ── CSS minimaliste ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    .disclaimer {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    .result-normal {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-pneumonia {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Avertissement médical permanent ──────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Avertissement médical</strong> — Ceci est un outil pédagogique,
pas un dispositif médical. Les résultats affichés <strong>ne remplacent
pas un avis médical professionnel</strong>. Ne prenez aucune décision
clinique sur la base de cet outil.
</div>
""", unsafe_allow_html=True)


# ── Titre ─────────────────────────────────────────────────────────────────────

st.title("🫁 Chest X-Ray Diagnostic Assistant")
st.caption(
    "Classification NORMAL / PNEUMONIA par ResNet-18 avec explicabilité Grad-CAM."
)

st.divider()


# ── Chargement du modèle (mis en cache pour toute la session) ─────────────────

@st.cache_resource(show_spinner="Chargement du modèle ResNet-18…")
def _load_cached_model():
    return load_model(_MODEL_PATH)


try:
    model = _load_cached_model()
except FileNotFoundError as e:
    st.error(
        f"❌ **Modèle introuvable**\n\n{e}\n\n"
        "Vérifiez que `notebooks/best_resnet18.pt` est présent dans le projet."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ **Erreur au chargement du modèle** : {e}")
    st.stop()


# ── Upload ────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📂 Charger une radiographie thoracique",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPEG, PNG",
)

if uploaded_file is None:
    st.info("⬆️ Uploadez une radiographie pour démarrer l'analyse.")
    st.stop()


# ── Validation de l'image ─────────────────────────────────────────────────────

try:
    image = Image.open(uploaded_file)
    image.verify()                     # détecte les fichiers corrompus
    uploaded_file.seek(0)              # rewind après verify()
    image = Image.open(uploaded_file)  # réouverture pour l'utilisation réelle
except UnidentifiedImageError:
    st.error("❌ Le fichier uploadé n'est pas une image valide (JPEG/PNG attendu).")
    st.stop()
except Exception as e:
    st.error(f"❌ Impossible de lire l'image : {e}")
    st.stop()


# ── Bouton d'analyse ──────────────────────────────────────────────────────────

if not st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True):
    # Aperçu de l'image avant analyse
    col_preview, _ = st.columns([1, 2])
    with col_preview:
        st.image(image, caption="Image uploadée", use_container_width=True)
    st.stop()


# ── Inférence ─────────────────────────────────────────────────────────────────

with st.spinner("Analyse en cours…"):
    try:
        result = predict(model, image)
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        st.stop()

    try:
        gradcam_img = compute_gradcam(model, result["tensor"], result["pred_idx"])
    except Exception as e:
        gradcam_img = None
        gradcam_error = str(e)
    else:
        gradcam_error = None


# ── Affichage du résultat ─────────────────────────────────────────────────────

predicted_class = result["predicted_class"]
confidence      = result["confidence"]
is_pneumonia    = predicted_class == "PNEUMONIA"

# Classe et score
css_class = "result-pneumonia" if is_pneumonia else "result-normal"
emoji     = "🔴" if is_pneumonia else "🟢"

st.markdown(
    f'<div class="{css_class}">'
    f'{emoji} Résultat : <strong>{predicted_class}</strong>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("#### Score de confiance")

# Barre de progression colorée
bar_color = "#dc3545" if is_pneumonia else "#28a745"
pct       = int(confidence * 100)

st.markdown(f"""
<div style="background:#e9ecef; border-radius:8px; height:22px; width:100%;">
  <div style="background:{bar_color}; width:{pct}%; height:22px;
              border-radius:8px; display:flex; align-items:center;
              padding-left:8px; color:white; font-size:0.85rem; font-weight:600;">
    {pct}%
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
| Classe | Probabilité |
|--------|------------|
| NORMAL | {result['probabilities']['NORMAL']*100:.1f}% |
| PNEUMONIA | {result['probabilities']['PNEUMONIA']*100:.1f}% |
""")

st.divider()

# ── Visualisation côte à côte ─────────────────────────────────────────────────

st.markdown("#### 🖼️ Image originale vs Heatmap Grad-CAM")

col_orig, col_cam = st.columns(2)

with col_orig:
    st.image(
        image.convert("RGB"),
        caption="Image originale",
        use_container_width=True,
    )

with col_cam:
    if gradcam_img is not None:
        st.image(
            gradcam_img,
            caption="Heatmap Grad-CAM — zones d'attention du modèle",
            use_container_width=True,
        )
    else:
        st.warning(f"Grad-CAM non disponible : {gradcam_error}")

st.divider()

# ── Note pédagogique sur Grad-CAM ─────────────────────────────────────────────

with st.expander("ℹ️ Comment lire la heatmap Grad-CAM ?"):
    st.markdown("""
La heatmap met en évidence les zones de la radiographie qui ont le plus
influencé la décision du modèle :

- **Rouge / chaud** → forte activation (le modèle a « regardé » cette zone)
- **Bleu / froid**  → faible activation

Dans le cas d'une pneumonie bactérienne, l'attention se concentre souvent
sur les condensations lobaires (bas des poumons). Pour une pneumonie virale,
l'attention est plus diffuse (opacités interstitielles bilatérales).

Cette visualisation ne valide pas le diagnostic — elle aide à comprendre
le raisonnement du modèle.
""")
