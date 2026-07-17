"""
Personalized Fashion Adviser - Streamlit Application.

Enhanced with deep learning:
- Multi-modal recommendations (visual + style + LSTM sequence prediction)
- Fashion category & attribute classification
- Diversity-aware results (MMR re-ranking)
- Complementary item suggestions
- Interactive filtering by category, style, season, color
- User session tracking for LSTM-based preference learning
"""

import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image
from numpy.linalg import norm

from models.feature_extractor import FashionFeatureExtractor
from models.fashion_classifier import FashionClassifier
from models.style_encoder import StyleEncoder
from models.autoencoder import FashionAutoencoder
from recommendation.engine import RecommendationEngine, SequencePredictor
from recommendation.diversity import DiversityReranker
from recommendation.filters import AttributeFilter


# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Adviser - AI Powered",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Personalized Fashion Adviser")
st.markdown(
    "**Deep Learning Powered** — Visual similarity, style matching, "
    "LSTM sequence prediction, and diversity-aware recommendations"
)


# ─────────────────────────────────────────────────────────────
# Load Data & Models (cached for performance)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    """Load all precomputed embeddings and metadata."""
    data = {}

    # Visual embeddings (required)
    data['visual'] = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    data['filenames'] = pickle.load(open('filenames.pkl', 'rb'))

    # Style embeddings (optional)
    if os.path.exists('style_embeddings.pkl'):
        data['style'] = np.array(pickle.load(open('style_embeddings.pkl', 'rb')))
    else:
        data['style'] = None

    # Refined embeddings (optional)
    if os.path.exists('refined_embeddings.pkl'):
        data['refined'] = np.array(pickle.load(open('refined_embeddings.pkl', 'rb')))
    else:
        data['refined'] = None

    # Metadata (optional)
    if os.path.exists('metadata.pkl'):
        data['metadata'] = pickle.load(open('metadata.pkl', 'rb'))
    else:
        data['metadata'] = None

    # Categories (optional)
    if os.path.exists('categories.pkl'):
        data['categories'] = pickle.load(open('categories.pkl', 'rb'))
    else:
        data['categories'] = None

    return data


@st.cache_resource
def load_models():
    """Initialize deep learning models."""
    models = {}

    # Feature extractor (ResNet50 for fast inference in app)
    models['extractor'] = FashionFeatureExtractor(
        embedding_dim=512, backbone='resnet50', fine_tune_layers=0
    )

    # Fashion classifier
    models['classifier'] = FashionClassifier()

    # Style encoder
    models['style_encoder'] = StyleEncoder(embedding_dim=256)

    return models


@st.cache_resource
def build_recommendation_engine(_data):
    """Build the multi-modal recommendation engine."""
    engine = RecommendationEngine(
        visual_weight=0.35,
        style_weight=0.25,
        category_weight=0.15,
        sequence_weight=0.25,
        n_neighbors=20,
        metric='cosine'
    )

    engine.build_index(
        visual_features=_data['visual'],
        style_features=_data['style'],
        refined_features=_data['refined'],
        categories=_data['categories'],
        filenames=_data['filenames']
    )

    return engine


# Load everything
data = load_embeddings()
models = load_models()
engine = build_recommendation_engine(data)
reranker = DiversityReranker(lambda_param=0.6)
attr_filter = AttributeFilter()

# Set up attribute filter with metadata
if data['metadata'] is not None:
    attr_filter.set_metadata(data['metadata'])

# Initialize session state for interaction history (LSTM)
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file):
    """Save uploaded image to disk."""
    try:
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', uploaded_file.name)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return filepath
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def extract_all_features(img_path):
    """Extract visual, style, and classification features for a query image."""
    # Visual embedding
    visual_features = models['extractor'].extract_features(img_path)

    # Style embedding
    style_features = models['style_encoder'].encode_style(img_path)

    # Classification
    classification = models['classifier'].predict(img_path)

    return visual_features, style_features, classification


def display_classification(classification):
    """Display the AI classification results."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Category",
            classification['category'],
            f"{classification['category_confidence']:.0%} confidence"
        )

    with col2:
        st.metric("Style", ", ".join(classification['styles']))

    with col3:
        st.metric("Season", ", ".join(classification['seasons']))

    with col4:
        st.metric("Colors", ", ".join(classification['colors'][:3]))


def display_recommendations(indices, title="Recommended Items", cols=6):
    """Display recommendation grid."""
    if not indices:
        st.info("No matching items found.")
        return

    st.subheader(title)
    columns = st.columns(min(cols, len(indices)))

    for i, idx in enumerate(indices[:cols]):
        with columns[i % cols]:
            if idx < len(data['filenames']):
                filepath = data['filenames'][idx]
                if os.path.exists(filepath):
                    st.image(filepath, use_container_width=True)
                    # Show category if available
                    if data['metadata'] is not None and idx < len(data['metadata']):
                        meta = data['metadata'][idx]
                        st.caption(f"{meta['category']} | {', '.join(meta['styles'][:2])}")
                else:
                    st.warning(f"Image not found")


# ─────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Recommendation Settings")

# Mode selection
rec_mode = st.sidebar.radio(
    "Recommendation Mode",
    ["Smart (Multi-modal)", "Visual Only", "Style Match", "Complementary Items"],
    help="Smart combines all signals. Visual uses appearance only. "
         "Style Match finds similar style regardless of look. "
         "Complementary suggests items that pair well."
)

# Number of results
n_results = st.sidebar.slider("Number of Results", 4, 12, 6)

# Diversity control
diversity = st.sidebar.slider(
    "Diversity Level",
    0.0, 1.0, 0.6,
    help="Higher = more varied results, Lower = more similar results"
)
reranker.lambda_param = 1.0 - diversity  # Invert for UX clarity

# Filters
st.sidebar.header("🔍 Filters")
filter_category = st.sidebar.selectbox(
    "Category Filter",
    ["None"] + [
        'Topwear', 'Bottomwear', 'Shoes', 'Bags', 'Accessories',
        'Dress', 'Sandals', 'Watches', 'Saree', 'Jewellery'
    ]
)
filter_style = st.sidebar.multiselect(
    "Style Filter",
    ['Casual', 'Formal', 'Sports', 'Ethnic', 'Party',
     'Smart Casual', 'Travel', 'Streetwear']
)
filter_season = st.sidebar.multiselect(
    "Season Filter",
    ['Spring', 'Summer', 'Autumn', 'Winter', 'All-Season']
)


# ─────────────────────────────────────────────────────────────
# Main Upload & Recommendation Flow
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a fashion image to get personalized recommendations",
    type=['jpg', 'jpeg', 'png', 'webp', 'bmp']
)

if uploaded_file is not None:
    filepath = save_uploaded_file(uploaded_file)

    if filepath:
        # Display uploaded image
        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(Image.open(uploaded_file), caption="Your Upload", width=300)

        with col_info:
            with st.spinner("🧠 Analyzing with deep learning models..."):
                visual_features, style_features, classification = \
                    extract_all_features(filepath)

            st.subheader("🤖 AI Analysis")
            display_classification(classification)

            # Add to interaction history for LSTM
            st.session_state.interaction_history.append(visual_features)
            # Keep last 20 interactions
            if len(st.session_state.interaction_history) > 20:
                st.session_state.interaction_history = \
                    st.session_state.interaction_history[-20:]

        st.divider()

        # Generate recommendations based on selected mode
        with st.spinner("Finding the best matches..."):

            if rec_mode == "Smart (Multi-modal)":
                # Full multi-modal recommendation with LSTM sequence awareness
                results = engine.recommend(
                    query_visual=visual_features,
                    query_style=style_features,
                    query_category=classification['category'],
                    interaction_history=(
                        st.session_state.interaction_history
                        if len(st.session_state.interaction_history) > 1
                        else None
                    ),
                    sequence_predictor=None,  # Use trained predictor if available
                    n_results=n_results * 3,  # Get extra for filtering/reranking
                    category_filter=(
                        filter_category if filter_category != "None" else None
                    )
                )
                candidate_indices = results['indices']
                candidate_scores = results['scores']

            elif rec_mode == "Visual Only":
                candidate_indices = list(
                    engine.recommend_simple(visual_features, n_results=n_results * 3)
                )
                candidate_scores = [1.0 / (i + 1) for i in range(len(candidate_indices))]

            elif rec_mode == "Style Match":
                candidate_indices = list(
                    engine.get_similar_by_style(style_features, n_results=n_results * 3)
                )
                candidate_scores = [1.0 / (i + 1) for i in range(len(candidate_indices))]

            elif rec_mode == "Complementary Items":
                candidate_indices = engine.get_complementary(
                    classification['category'], style_features, n_results=n_results * 3
                )
                candidate_scores = [1.0 / (i + 1) for i in range(len(candidate_indices))]

            # Apply attribute filters
            if filter_style:
                candidate_indices = attr_filter.filter_by_style(
                    candidate_indices, filter_style
                )
            if filter_season:
                candidate_indices = attr_filter.filter_by_season(
                    candidate_indices, filter_season
                )

            # Apply diversity re-ranking
            if len(candidate_indices) > n_results and len(candidate_scores) >= len(candidate_indices):
                scores_for_rerank = candidate_scores[:len(candidate_indices)]
                final_indices = reranker.rerank(
                    candidate_indices,
                    scores_for_rerank,
                    data['visual'],
                    n_results=n_results
                )
            else:
                final_indices = candidate_indices[:n_results]

        # Display results
        display_recommendations(final_indices, "✨ Recommended Items", cols=n_results)

        # Show complementary items if in Smart mode
        if rec_mode == "Smart (Multi-modal)":
            st.divider()
            complementary = engine.get_complementary(
                classification['category'], style_features, n_results=4
            )
            if complementary:
                display_recommendations(
                    complementary,
                    "🎯 Complete the Look (Complementary Items)",
                    cols=4
                )

        # Show diversity score
        if final_indices:
            div_score = reranker.compute_diversity_score(
                final_indices, data['visual']
            )
            st.sidebar.metric("Recommendation Diversity", f"{div_score:.2f}")

        # Session info
        st.sidebar.divider()
        st.sidebar.caption(
            f"📊 Session: {len(st.session_state.interaction_history)} interactions tracked"
        )
        if st.sidebar.button("Clear History"):
            st.session_state.interaction_history = []
            st.rerun()

    else:
        st.error("Failed to save the uploaded file. Please try again.")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 0.8em;'>
    <p>Powered by ResNet50 + LSTM + Triplet Loss + Variational Autoencoder</p>
    <p>Multi-modal recommendations with diversity-aware re-ranking</p>
</div>
""", unsafe_allow_html=True)
