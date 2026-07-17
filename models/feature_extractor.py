"""
Enhanced Feature Extraction Module.

Uses a fine-tunable ResNet50 backbone with additional dense layers
for generating rich fashion-aware embeddings. Supports both frozen
(pretrained) and fine-tuned modes.
"""

import numpy as np
import tensorflow as tf
import keras.ops as ops
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    GlobalMaxPooling2D, GlobalAveragePooling2D, Dense, Dropout,
    BatchNormalization, Concatenate, Input, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm


class FashionFeatureExtractor:
    """
    Enhanced feature extractor that combines:
    1. ResNet50 for structural features
    2. EfficientNetB3 for texture/pattern features
    3. Multi-scale feature fusion
    4. Learnable projection head for compact embeddings
    """

    def __init__(self, embedding_dim=512, backbone='dual', fine_tune_layers=20):
        """
        Args:
            embedding_dim: Dimension of the output embedding vector
            backbone: 'resnet50', 'efficientnet', or 'dual' for ensemble
            fine_tune_layers: Number of top backbone layers to unfreeze for fine-tuning
        """
        self.embedding_dim = embedding_dim
        self.backbone_type = backbone
        self.fine_tune_layers = fine_tune_layers
        self.model = self._build_model()

    def _build_model(self):
        """Build the feature extraction model with projection head."""
        input_tensor = Input(shape=(224, 224, 3))

        if self.backbone_type == 'dual':
            return self._build_dual_backbone(input_tensor)
        elif self.backbone_type == 'efficientnet':
            return self._build_efficientnet(input_tensor)
        else:
            return self._build_resnet(input_tensor)

    def _build_resnet(self, input_tensor):
        """ResNet50 backbone with projection head."""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )

        # Freeze early layers, allow fine-tuning of top layers
        for layer in base_model.layers[:-self.fine_tune_layers]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = self._projection_head(x)

        model = Model(inputs=input_tensor, outputs=x, name='resnet_extractor')
        return model

    def _build_efficientnet(self, input_tensor):
        """EfficientNetB3 backbone - better at texture/pattern recognition."""
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )

        for layer in base_model.layers[:-self.fine_tune_layers]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = self._projection_head(x)

        model = Model(inputs=input_tensor, outputs=x, name='efficientnet_extractor')
        return model

    def _build_dual_backbone(self, input_tensor):
        """
        Dual backbone architecture: combines ResNet50 (structural features)
        and EfficientNetB3 (texture features) for richer representations.
        """
        # ResNet50 branch - structural features
        resnet = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        for layer in resnet.layers[:-self.fine_tune_layers]:
            layer.trainable = False
        # Rename layers to avoid conflicts
        for layer in resnet.layers:
            layer._name = 'resnet_' + layer.name

        resnet_features = resnet(input_tensor)
        resnet_pooled = GlobalAveragePooling2D(name='resnet_gap')(resnet_features)

        # EfficientNet branch - texture/pattern features
        efficientnet = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        for layer in efficientnet.layers[:-self.fine_tune_layers]:
            layer.trainable = False
        for layer in efficientnet.layers:
            layer._name = 'effnet_' + layer.name

        effnet_features = efficientnet(input_tensor)
        effnet_pooled = GlobalAveragePooling2D(name='effnet_gap')(effnet_features)

        # Multi-scale feature fusion
        combined = Concatenate(name='feature_fusion')([resnet_pooled, effnet_pooled])

        # Shared projection head
        x = self._projection_head(combined)

        model = Model(inputs=input_tensor, outputs=x, name='dual_extractor')
        return model

    def _projection_head(self, x):
        """
        Learnable projection head that maps backbone features
        to a compact, normalized embedding space.
        """
        x = Dense(1024, activation='relu', name='proj_dense1')(x)
        x = BatchNormalization(name='proj_bn1')(x)
        x = Dropout(0.3, name='proj_dropout1')(x)
        x = Dense(self.embedding_dim, activation='relu', name='proj_dense2')(x)
        x = BatchNormalization(name='proj_bn2')(x)
        # L2 normalization for cosine similarity compatibility
        x = Lambda(lambda t: ops.normalize(t, axis=1), name='l2_norm')(x)
        return x

    def extract_features(self, img_path):
        """
        Extract normalized feature embedding from a single image.

        Args:
            img_path: Path to the image file

        Returns:
            Normalized feature vector of shape (embedding_dim,)
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = self.model.predict(preprocessed_img, verbose=0).flatten()
        normalized_result = result / (norm(result) + 1e-7)
        return normalized_result

    def extract_features_batch(self, img_paths, batch_size=32):
        """
        Extract features from multiple images in batches for efficiency.

        Args:
            img_paths: List of image file paths
            batch_size: Number of images to process at once

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        all_features = []

        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = image.load_img(path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    batch_images.append(img_array)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
                    # Use zero image as placeholder
                    batch_images.append(np.zeros((224, 224, 3)))

            batch_array = np.array(batch_images)
            preprocessed_batch = preprocess_input(batch_array)
            features = self.model.predict(preprocessed_batch, verbose=0)

            # Normalize each feature vector
            norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-7
            normalized_features = features / norms
            all_features.append(normalized_features)

        return np.vstack(all_features)

    def save_model(self, path='saved_models/feature_extractor'):
        """Save the model weights."""
        self.model.save_weights(path)

    def load_model(self, path='saved_models/feature_extractor'):
        """Load pretrained model weights."""
        self.model.load_weights(path)

    def get_model(self):
        """Return the underlying Keras model for training."""
        return self.model
