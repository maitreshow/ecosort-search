"""
Jalon 2b - Application Streamlit EcoSort-Search.
Auteur: ARZIKA

Flux :
    1. L'utilisateur tape un mot-cle
    2. On cherche sur Jumia (scraping/jumia_scraper.py)
    3. L'utilisateur choisit un produit parmi les resultats
    4. Le modele (model/modele_eco_sort.h5) predit la matiere
    5. On mappe la matiere vers la bonne poubelle et on colore l'ecran

Usage :
    streamlit run app/app.py
"""

import sys
import os

# Permet d'importer le module scraping/ depuis app/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from scraping.jumia_scraper import search_jumia

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "modele_eco_sort.h5")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "class_names.txt")
IMG_SIZE = (224, 224)

# Mapping : classe predite par le modele -> categorie officielle de tri
CATEGORY_MAPPING = {
    "cardboard": "JAUNE",
    "plastic": "JAUNE",
    "metal": "JAUNE",
    "glass": "VERTE",
    "paper": "BLEUE",
    "electronic": "D3E",
    "trash": "MARRON",
}

# Couleurs et infos d'affichage par categorie officielle
CATEGORY_INFO = {
    "JAUNE": {
        "couleur": "#FFD700",
        "emoji": "🟡",
        "nom": "Poubelle JAUNE",
        "description": "Emballages menagers legers : plastique, metal, carton",
    },
    "VERTE": {
        "couleur": "#2E8B57",
        "emoji": "🟢",
        "nom": "Poubelle VERTE",
        "description": "Verre d'emballage uniquement",
    },
    "BLEUE": {
        "couleur": "#1E90FF",
        "emoji": "🔵",
        "nom": "Poubelle BLEUE",
        "description": "Papiers graphiques propres",
    },
    "D3E": {
        "couleur": "#808080",
        "emoji": "🎛️",
        "nom": "Bac Electronique (D3E)",
        "description": "Piles, batteries, appareils electriques",
    },
    "MARRON": {
        "couleur": "#5C4033",
        "emoji": "⚫",
        "nom": "Poubelle MARRON/NOIRE",
        "description": "Dechets residuels non recyclables",
    },
}

# ---------------------------------------------------------------------------
# CHARGEMENT DU MODELE (une seule fois, mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def predict_category(image: Image.Image, model, class_names):
    """Predit la classe d'une image PIL et retourne (classe, confiance)."""
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[predicted_idx])

    return predicted_class, confidence


def download_image(url: str) -> Image.Image | None:
    """Telecharge une image depuis une URL et la retourne en objet PIL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Impossible de charger l'image : {e}")
        return None


# ---------------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EcoSort-Search", page_icon="♻️", layout="centered")

st.title("♻️ EcoSort-Search")
st.markdown("Recherchez un produit, l'IA vous dira dans quelle poubelle le jeter.")

# Initialisation de l'etat de session (pour garder les resultats entre les clics)
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_result" not in st.session_state:
    st.session_state.selected_result = None

# --- Barre de recherche ---
keyword = st.text_input("Nom du produit (ex : bouteille shampoing, smartphone, journal...)")

if st.button("🔍 Rechercher sur Jumia") and keyword:
    with st.spinner("Recherche en cours sur Jumia..."):
        st.session_state.search_results = search_jumia(keyword, max_results=5)
        st.session_state.selected_result = None

# --- Affichage des resultats ---
if st.session_state.search_results:
    st.subheader("Resultats trouves")

    cols = st.columns(len(st.session_state.search_results))
    for i, produit in enumerate(st.session_state.search_results):
        with cols[i]:
            if produit["image_url"]:
                st.image(produit["image_url"], width="stretch")
            st.caption(produit["nom"][:60])
            st.caption(produit["prix"])
            if st.button("Choisir", key=f"select_{i}"):
                st.session_state.selected_result = produit

elif keyword and st.session_state.search_results == []:
    st.warning("Aucun produit trouve. Essayez un autre mot-cle.")

# --- Prediction sur le produit selectionne ---
if st.session_state.selected_result:
    produit = st.session_state.selected_result
    st.divider()
    st.subheader(f"Analyse de : {produit['nom']}")

    model, class_names = load_model()

    with st.spinner("Analyse de l'image en cours..."):
        image = download_image(produit["image_url"])

        if image:
            predicted_class, confidence = predict_category(image, model, class_names)
            categorie = CATEGORY_MAPPING.get(predicted_class, "MARRON")
            info = CATEGORY_INFO[categorie]

            # Affichage colore du resultat
            st.markdown(
                f"""
                <div style="
                    background-color: {info['couleur']};
                    padding: 40px;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                ">
                    {info['emoji']} {info['nom']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"**Matiere detectee :** {predicted_class} ({confidence*100:.1f}% de confiance)")
            st.markdown(f"**Consigne :** {info['description']}")

            st.image(image, caption=produit["nom"], width=200)
