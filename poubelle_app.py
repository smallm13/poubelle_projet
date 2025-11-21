import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# -------------------------
# Configuration de la page
# -------------------------
st.set_page_config(
    page_title="Détection Poubelle Intelligente",
    page_icon="🗑️",
    layout="centered"
)

# -------------------------
# CSS personnalisé
# -------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .upload-section {
        border: 2px dashed #2E86AB;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #f8f9fa;
    }
    .result-badge {
        font-size: 1.3rem;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 1rem 0;
        display: inline-block;
    }
    .refresh-button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .refresh-button:hover {
        background-color: #1a5a7a;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 1️⃣ Charger le modèle
# -------------------------
MODEL_PATH = "poubelle_model_effnet.keras"

def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle : {e}")
            return None
    else:
        st.error(f"❌ Le fichier {MODEL_PATH} est introuvable.")
        return None

model = load_my_model()

# Classes binaires
class_names = ["pleine", "vide"]

# -------------------------
# Fonction pour réinitialiser
# -------------------------
def reset_app():
    st.session_state.clear()
    st.rerun()

# -------------------------
# Interface principale
# -------------------------
st.markdown('<div class="main-header">🗑️ Détection de Poubelle</div>', unsafe_allow_html=True)

# Section d'upload
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Téléversez une image de poubelle")
st.markdown("Formats supportés : JPG, JPEG, PNG")

uploaded_file = st.file_uploader(
    "Choisir une image...", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Traitement et prédiction
# -------------------------
if uploaded_file is not None and model is not None:
    # Afficher l'image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image analysée", use_container_width=True)

    with st.spinner('🔄 Analyse en cours...'):
        # Prétraitement de l'image
        img_array = np.array(img.resize((224, 224)))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Prédiction
        pred = model.predict(img_array, verbose=0)[0][0]
        confidence = pred if pred >= 0.5 else 1 - pred
        predicted_class = class_names[1] if pred >= 0.5 else class_names[0]

    # Affichage des résultats
    st.markdown("### 📊 Résultat")

    # Badge de résultat
    if predicted_class == "pleine":
        badge_color = "🔴"
        background_color = "#ff4444"
        recommendation = "🚨 Il est temps de vider la poubelle !"
    else:
        badge_color = "🟢" 
        background_color = "#00C851"
        recommendation = "✅ La poubelle peut encore être utilisée !"

    st.markdown(f"""
    <div style='text-align: center;'>
        <div class='result-badge' style='background-color: {background_color}; color: white;'>
            {badge_color} Poubelle {predicted_class.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Confiance : {confidence:.1%}**")
    st.markdown(f"<div class='confidence-fill' style='width: {confidence*100}%; background-color: {background_color};'></div>", unsafe_allow_html=True)

    st.markdown(f"**Score :** `{pred:.3f}`")
    st.info(recommendation)

    # Bouton de réinitialisation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button(
            "🔄 Nouvelle analyse", 
            on_click=reset_app,
            use_container_width=True,
            type="primary"
        )

elif uploaded_file is not None and model is None:
    st.error("❌ Le modèle n'est pas disponible.")

# -------------------------
# Bouton de téléchargement du modèle
# -------------------------
st.markdown("---")
st.markdown("### 📁 Télécharger le modèle")
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        st.download_button(
            label="📥 Télécharger le modèle IA",
            data=file,
            file_name="poubelle_model_effnet.keras",
            mime="application/octet-stream",
            help="Téléchargez le modèle pour l'utiliser localement",
            use_container_width=True
        )
else:
    st.warning("Modèle non disponible pour le téléchargement")

# -------------------------
# Informations techniques
# -------------------------
with st.expander("ℹ️ Informations techniques"):
    st.markdown("""
    **Comment utiliser :**
    1. 📸 Prenez une photo de votre poubelle
    2. ⬆️ Téléversez l'image
    3. 🤖 L'IA détecte si elle est vide ou pleine

    **Spécifications :**
    - Architecture : EfficientNetB0
    - Type : Classification binaire
    - Taille d'entrée : 224x224 pixels
    - Format : .keras (TensorFlow)
    """)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Détection intelligente de poubelles | Streamlit & TensorFlow"
    "</div>", 
    unsafe_allow_html=True
)