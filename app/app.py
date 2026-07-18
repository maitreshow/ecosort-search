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

# Mots-cles indiquant qu'un produit est reellement electronique.
# Utilise comme garde-fou : si le modele predit "electronic" sans etre tres
# confiant ET qu'aucun de ces mots n'apparait dans le nom du produit,
# on ne fait pas confiance a cette prediction (biais connu du modele,
# qui associe parfois a tort "fond uni / couleur unie" a "electronic").
ELECTRONIC_KEYWORDS = [
    "smartphone", "telephone", "téléphone", "phone", "chargeur", "ecouteur",
    "écouteur", "casque", "mixeur", "montre", "watch", "tablette", "tablet",
    "ordinateur", "laptop", "pc", "clavier", "souris", "imprimante",
    "television", "télévision", "tv", "radio", "camera", "caméra",
    "batterie", "pile", "cable", "câble", "adaptateur", "haut-parleur",
    "enceinte", "micro-ondes", "refrigerateur", "réfrigérateur",
    "climatiseur", "ventilateur", "rasoir", "seche-cheveux", "fer a repasser",
]

# Seuil de confiance en dessous duquel on se mefie d'une prediction "electronic"
ELECTRONIC_CONFIDENCE_THRESHOLD = 0.80


def is_likely_electronic_by_name(product_name: str) -> bool:
    """Verifie si le nom du produit contient un mot-cle electronique evident."""
    name_lower = product_name.lower()
    return any(keyword in name_lower for keyword in ELECTRONIC_KEYWORDS)


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
    """
    Predit la classe d'une image PIL.
    Retourne une liste de (classe, confiance) triee par confiance decroissante,
    pour permettre un garde-fou (ex: se rabattre sur le 2e choix si besoin).
    """
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    ranked = sorted(
        zip(class_names, predictions.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked  # ex: [("electronic", 0.62), ("plastic", 0.25), ...]


def apply_electronic_guard(ranked_predictions, product_name: str):
    """
    Garde-fou : si la meilleure prediction est 'electronic' avec une confiance
    moyenne (pas tres sure) ET qu'aucun mot-cle electronique n'apparait dans
    le nom du produit, on se rabat sur la 2e meilleure prediction.

    Corrige un biais connu du modele qui associe parfois a tort les objets
    a couleur unie / fond neutre a la classe "electronic".
    """
    best_class, best_confidence = ranked_predictions[0]

    if (
        best_class == "electronic"
        and best_confidence < ELECTRONIC_CONFIDENCE_THRESHOLD
        and not is_likely_electronic_by_name(product_name)
        and len(ranked_predictions) > 1
    ):
        # On se rabat sur le 2e choix, et on garde une trace pour l'affichage
        second_class, second_confidence = ranked_predictions[1]
        return second_class, second_confidence, True  # True = garde-fou active

    return best_class, best_confidence, False


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
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

st.set_page_config(
    page_title="EcoSort-Search",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "♻️",
    layout="centered",
)

# --- CSS personnalise : police, couleurs de marque, cartes produits ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Bandeau d'en-tete */
    .ecosort-header {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 10px 0 20px 0;
    }
    .ecosort-header h1 {
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0;
        background: linear-gradient(90deg, #1B5E20, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ecosort-header p {
        margin: 2px 0 0 0;
        color: #6b7280;
        font-size: 0.95rem;
    }

    /* Carte produit */
    .product-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .product-card:hover {
        box-shadow: 0 6px 16px rgba(27,94,32,0.15);
        transform: translateY(-3px);
    }
    .product-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1f2937;
        min-height: 40px;
        margin-top: 6px;
    }
    .product-price {
        color: #2E7D32;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    /* Boutons Streamlit */
    div.stButton > button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #1B5E20;
        color: white;
    }

    /* Bandeau resultat colore : coins arrondis + ombre douce */
    .result-banner {
        padding: 36px;
        border-radius: 18px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- En-tete avec logo ---
header_cols = st.columns([1, 6])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=70)
with header_cols[1]:
    st.markdown(
        """
        <div class="ecosort-header">
            <div>
                <h1>EcoSort-Search</h1>
                <p>Recherchez un produit, l'IA vous dira dans quelle poubelle le jeter ♻️</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialisation de l'etat de session (pour garder les resultats entre les clics)
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_result" not in st.session_state:
    st.session_state.selected_result = None

# --- Barre de recherche ---
keyword = st.text_input(
    "Nom du produit",
    placeholder="ex : bouteille shampoing, smartphone, journal...",
    label_visibility="collapsed",
)

if st.button("🔍  Rechercher sur Jumia") and keyword:
    with st.spinner("Recherche en cours sur Jumia..."):
        st.session_state.search_results = search_jumia(keyword, max_results=5)
        st.session_state.selected_result = None

# --- Affichage des resultats ---
if st.session_state.search_results:
    st.markdown("#### Résultats trouvés")

    cols = st.columns(len(st.session_state.search_results))
    for i, produit in enumerate(st.session_state.search_results):
        with cols[i]:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            if produit["image_url"]:
                st.image(produit["image_url"], width="stretch")
            st.markdown(
                f'<div class="product-name">{produit["nom"][:55]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="product-price">{produit["prix"]}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Choisir", key=f"select_{i}"):
                st.session_state.selected_result = produit
            st.markdown("</div>", unsafe_allow_html=True)

elif keyword and st.session_state.search_results == []:
    st.warning("Aucun produit trouvé. Essayez un autre mot-clé.")

# --- Prediction sur le produit selectionne ---
if st.session_state.selected_result:
    produit = st.session_state.selected_result
    st.divider()
    st.markdown(f"#### Analyse de : {produit['nom']}")

    model, class_names = load_model()

    with st.spinner("Analyse de l'image en cours..."):
        image = download_image(produit["image_url"])

        if image:
            ranked_predictions = predict_category(image, model, class_names)
            predicted_class, confidence, guard_triggered = apply_electronic_guard(
                ranked_predictions, produit["nom"]
            )
            categorie = CATEGORY_MAPPING.get(predicted_class, "MARRON")
            info = CATEGORY_INFO[categorie]

            # Affichage colore du resultat
            st.markdown(
                f"""
                <div class="result-banner" style="background-color: {info['couleur']};">
                    {info['emoji']} {info['nom']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            result_cols = st.columns([1, 2])
            with result_cols[0]:
                st.image(image, width="stretch")
            with result_cols[1]:
                st.markdown(f"**Matière détectée :** {predicted_class}")
                st.markdown(f"**Confiance :** {confidence*100:.1f}%")
                st.markdown(f"**Consigne :** {info['description']}")

                if guard_triggered:
                    st.caption(
                        "ℹ️ Le modèle hésitait avec 'electronic' (confiance moyenne, "
                        "pas de mot-clé électronique dans le nom du produit) : "
                        "la 2e prédiction la plus probable a été retenue."
                    )

            with st.expander("Voir le détail des probabilités par classe"):
                for cls, conf in ranked_predictions:
                    st.progress(conf, text=f"{cls} — {conf*100:.1f}%")
