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
</style>
""", unsafe_allow_html=True)

# -------------------------
# Charger le modèle
# -------------------------
MODEL_PATH = "poubelle_model_effnet.keras"

@st.cache_resource
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

model = load_my_model()
class_names = ["pleine", "vide"]

# -------------------------
# Interface principale
# -------------------------
st.markdown('<div class="main-header">🗑️ Détection de Poubelle</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Téléversez une image de poubelle")
st.markdown("Formats supportés : JPG, JPEG, PNG")

uploaded_file = st.file_uploader(
    "Choisir une image...",
    type=["jpg", "jpeg", "png"],
    key="uploaded"
)
st.markdown('</div>', unsafe_allow_html=True)

# Bouton analyse
analyze_button = st.button("🟢 Analyser l'image")

# -------------------------
# Analyse
# -------------------------
if uploaded_file is not None and analyze_button and model is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image analysée", use_container_width=True)

    with st.spinner('🔄 Analyse en cours...'):
        img_processed = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_processed), axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        pred = model.predict(img_array, verbose=0)[0][0]
        confidence = pred if pred >= 0.5 else 1 - pred
        predicted_class = class_names[1] if pred >= 0.5 else class_names[0]

    # Résultat
    if predicted_class == "pleine":
        color = "#ff4444"
        icon = "🔴"
        recommendation = "🚨 La poubelle est pleine !"
    else:
        color = "#00C851"
        icon = "🟢"
        recommendation = "✅ La poubelle est encore utilisable."

    st.markdown(f"""
    <div style='text-align:center; margin-top:20px;'>
        <span class='result-badge' style='background:{color}; color:white;'>
            {icon} Poubelle {predicted_class.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 📊 Confiance : **{confidence:.1%}**")
    st.info(recommendation)

    # 🔄 Bouton pour réactualiser
    if st.button("🔄 Réinitialiser la page"):
        st.session_state.clear()
        st.experimental_rerun()

elif uploaded_file is not None and model is None:
    st.error("❌ Le modèle n'est pas disponible.")

# -------------------------
# Bouton de téléchargement
# -------------------------
st.markdown("---")
st.markdown("### 📁 Télécharger le modèle")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        st.download_button(
            "📥 Télécharger le modèle IA",
            file,
            file_name="poubelle_model_effnet.keras",
            mime="application/octet-stream"
        )
else:
    st.warning("Modèle non disponible.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>Détection intelligente de poubelles | Streamlit & TensorFlow</div>",
    unsafe_allow_html=True
)
