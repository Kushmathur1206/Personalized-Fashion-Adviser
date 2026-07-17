"""
Deep Learning Models for Personalized Fashion Adviser.

Modules:
- feature_extractor: Enhanced CNN feature extraction with fine-tuning support
- fashion_classifier: Category and attribute classification
- style_encoder: Triplet-loss based style embedding network
- autoencoder: Embedding refinement via variational autoencoder
"""

from models.feature_extractor import FashionFeatureExtractor
from models.fashion_classifier import FashionClassifier
from models.style_encoder import StyleEncoder
from models.autoencoder import FashionAutoencoder
