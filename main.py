"""
Personalized Fashion Adviser — Outfit Builder.

Upload a clothing item, pick an occasion, get a complete styled outfit.

Models used:
- ResNet50 + projection head: extracts style-aware embeddings
- LSTM-VAE: refines embeddings for better similarity matching
- Outfit rules: decides which slots to fill
- Cosine similarity: ranks items per slot by style match
"""

import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image as keras_image
from numpy.linalg import norm

from outfit_builder import (
    build_outfit, classify_upload, OCCASIONS, CATEGORY_MAP
)
from lstm_outfit import OutfitLSTM, generate_outfit_lstm


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fashion Outfit Builder", page_icon="👗", layout="wide")

SLOT_ICONS = {
    'top': '👕',
    'bottom': '👖',
    'shoes': '�',
    'outerwear': '🧥',
    'accessory': '🧢',
    'dress': '👗',
}

SLOT_NAMES = {
    'top': 'Top',
    'bottom': 'Bottom',
    'shoes': 'Shoes',
    'outerwear': 'Outerwear',
    'accessory': 'Accessory',
    'dress': 'Dress',
}


# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load feature extraction model (ResNet50 + pooling)."""
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
    ])
    return model


@st.cache_data
def load_catalog():
    """Load precomputed embeddings, labels, and filenames."""
    embeddings = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    labels = pickle.load(open('labels.pkl', 'rb'))
    return embeddings, filenames, labels


def extract_embedding(img_path, model):
    """Extract normalized embedding from an image."""
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0).flatten()
    return features / (norm(features) + 1e-7)


def predict_category(embedding, catalog_embeddings, catalog_labels):
    """
    Predict category of uploaded item by finding nearest neighbors
    and taking majority vote. Simple but effective.
    """
    # Cosine similarity to all catalog items
    sims = catalog_embeddings @ embedding
    top_k = np.argsort(sims)[-10:]  # top 10 most similar
    top_labels = [catalog_labels[i] for i in top_k]

    # Majority vote
    from collections import Counter
    votes = Counter(top_labels)
    return votes.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("👗 Outfit Builder")
st.markdown("Upload a clothing item → pick an occasion → get a complete outfit")

# Sidebar
st.sidebar.header("Occasion")
occasion = st.sidebar.radio(
    "What's the occasion?",
    list(OCCASIONS.keys()),
    index=0,
)
n_options = st.sidebar.slider("Options per slot", 2, 5, 3)

# Upload
uploaded_file = st.file_uploader(
    "Upload a clothing image", type=['jpg', 'jpeg', 'png', 'webp']
)

if uploaded_file is not None:
    # Save file
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', uploaded_file.name)
    with open(filepath, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Load resources
    model = load_model()
    catalog_embeddings, catalog_filenames, catalog_labels = load_catalog()

    # Extract embedding
    with st.spinner("Analyzing your item..."):
        query_embedding = extract_embedding(filepath, model)
        predicted_label = predict_category(
            query_embedding, catalog_embeddings, catalog_labels
        )

    # Show uploaded item
    uploaded_group = classify_upload(predicted_label)

    col_upload, col_info = st.columns([1, 2])
    with col_upload:
        st.image(Image.open(uploaded_file), width=250)
    with col_info:
        st.markdown(f"**Detected:** {predicted_label}")
        st.markdown(f"**Category:** {SLOT_NAMES.get(uploaded_group, uploaded_group)}")
        st.markdown(f"**Occasion:** {occasion}")

    st.divider()

    # Build outfit
    with st.spinner("Building your outfit..."):
        outfit = build_outfit(
            query_embedding=query_embedding,
            uploaded_label=predicted_label,
            occasion=occasion,
            catalog_embeddings=catalog_embeddings,
            catalog_labels=catalog_labels,
            catalog_filenames=catalog_filenames,
            n_options=n_options,
        )

    if not outfit:
        st.warning("Couldn't build an outfit — not enough items in catalog for this combination.")
    else:
        st.subheader("Your Outfit (Rule-based + Style Matching)")

        for slot, options in outfit.items():
            icon = SLOT_ICONS.get(slot, '🔹')
            name = SLOT_NAMES.get(slot, slot.title())
            st.markdown(f"### {icon} {name}")

            cols = st.columns(len(options))
            for i, item in enumerate(options):
                with cols[i]:
                    if os.path.exists(item['filename']):
                        st.image(item['filename'], use_container_width=True)
                        st.caption(f"{item['label']} • {item['score']:.0%} match")
                    else:
                        st.error("Image not found")

    # ─────────────────────────────────────────────────────────
    # LSTM Row
    # ─────────────────────────────────────────────────────────
    st.divider()

    lstm_path = 'saved_models/outfit_lstm.weights.h5'
    if os.path.exists(lstm_path):
        st.subheader("🧠 LSTM Suggestion (Sequential Prediction)")
        st.caption(
            "Each item is predicted based on ALL previous items — "
            "the shoes know what pants were picked."
        )

        lstm_model = OutfitLSTM()
        lstm_model.load(lstm_path)

        with st.spinner("LSTM generating outfit..."):
            lstm_results = generate_outfit_lstm(
                lstm_model=lstm_model,
                query_embedding=query_embedding,
                query_label=predicted_label,
                occasion=occasion,
                catalog_embeddings=catalog_embeddings,
                catalog_labels=catalog_labels,
                catalog_filenames=catalog_filenames,
                max_items=3,
            )

        if lstm_results:
            cols = st.columns(len(lstm_results))
            for i, item in enumerate(lstm_results):
                with cols[i]:
                    if os.path.exists(item['filename']):
                        st.image(item['filename'], use_container_width=True)
                        slot_icon = SLOT_ICONS.get(item['slot'], '🔹')
                        st.caption(
                            f"{slot_icon} {item['label']} • "
                            f"{item['score']:.0%} • Step {i+1}"
                        )
                    else:
                        st.error("Image not found")
        else:
            st.info("LSTM couldn't generate suggestions for this combination.")
    else:
        st.info("🧠 LSTM model not found. Run `python app.py` to train it.")

    st.divider()
    st.info("💡 Try switching the occasion — both rule-based and LSTM results will change.")
