"""
Style Encoder with Triplet Loss Training.

Learns a style-aware embedding space where:
- Similar style items are close together
- Different style items are far apart

Uses triplet loss (anchor, positive, negative) to learn
style similarity that goes beyond visual appearance.
"""

import numpy as np
import tensorflow as tf
import keras.ops as ops
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Input, Lambda, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm


class TripletLossLayer(tf.keras.layers.Layer):
    """Custom layer that computes triplet loss."""

    def __init__(self, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, inputs):
        anchor, positive, negative = inputs

        # Euclidean distances
        pos_dist = ops.sum(ops.square(anchor - positive), axis=1)
        neg_dist = ops.sum(ops.square(anchor - negative), axis=1)

        # Triplet loss with margin
        loss = ops.maximum(pos_dist - neg_dist + self.margin, 0.0)
        self.add_loss(ops.mean(loss))

        return anchor

    def get_config(self):
        config = super().get_config()
        config.update({'margin': self.margin})
        return config


class StyleEncoder:
    """
    Style-aware embedding network trained with triplet loss.

    The key insight: two items can look visually different but have
    similar style (e.g., a blue formal shirt and a black formal pants).
    This network learns to capture style similarity beyond pixel-level
    appearance.
    """

    def __init__(self, embedding_dim=256, margin=0.5):
        """
        Args:
            embedding_dim: Dimension of the style embedding
            margin: Triplet loss margin (controls separation between
                    positive and negative pairs)
        """
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.encoder = self._build_encoder()
        self.training_model = self._build_triplet_model()

    def _build_encoder(self):
        """
        Build the style encoder network.
        Maps an image to a style embedding vector.
        """
        input_tensor = Input(shape=(224, 224, 3), name='encoder_input')

        # Use ResNet50 as backbone
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        # Only fine-tune top layers
        for layer in base_model.layers[:-15]:
            layer.trainable = False

        x = base_model(input_tensor)
        x = GlobalAveragePooling2D()(x)

        # Style-specific projection
        x = Dense(512, activation='relu', name='style_dense1')(x)
        x = BatchNormalization(name='style_bn1')(x)
        x = Dropout(0.3, name='style_dropout1')(x)
        x = Dense(self.embedding_dim, activation='relu', name='style_dense2')(x)
        x = BatchNormalization(name='style_bn2')(x)

        # L2 normalize for angular distance
        x = Lambda(
            lambda t: ops.normalize(t, axis=1),
            name='style_l2_norm'
        )(x)

        encoder = Model(inputs=input_tensor, outputs=x, name='style_encoder')
        return encoder

    def _build_triplet_model(self):
        """
        Build the triplet training model.

        Takes three inputs (anchor, positive, negative) and minimizes
        the triplet loss to learn style-aware embeddings.
        """
        # Three input branches
        anchor_input = Input(shape=(224, 224, 3), name='anchor_input')
        positive_input = Input(shape=(224, 224, 3), name='positive_input')
        negative_input = Input(shape=(224, 224, 3), name='negative_input')

        # Shared encoder for all three
        anchor_embedding = self.encoder(anchor_input)
        positive_embedding = self.encoder(positive_input)
        negative_embedding = self.encoder(negative_input)

        # Triplet loss computation
        output = TripletLossLayer(margin=self.margin, name='triplet_loss')(
            [anchor_embedding, positive_embedding, negative_embedding]
        )

        model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=output,
            name='triplet_training_model'
        )
        return model

    def compile_model(self, learning_rate=1e-4):
        """Compile the triplet training model."""
        self.training_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=None  # Loss is computed in TripletLossLayer
        )

    def encode_style(self, img_path):
        """
        Extract style embedding from a single image.

        Args:
            img_path: Path to the image

        Returns:
            Style embedding vector of shape (embedding_dim,)
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        embedding = self.encoder.predict(preprocessed_img, verbose=0).flatten()
        return embedding

    def encode_style_batch(self, img_paths, batch_size=32):
        """
        Extract style embeddings from multiple images.

        Args:
            img_paths: List of image paths
            batch_size: Batch size for inference

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        all_embeddings = []

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
                    batch_images.append(np.zeros((224, 224, 3)))

            batch_array = np.array(batch_images)
            preprocessed_batch = preprocess_input(batch_array)
            embeddings = self.encoder.predict(preprocessed_batch, verbose=0)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def compute_style_similarity(self, embedding1, embedding2):
        """
        Compute style similarity between two embeddings.

        Args:
            embedding1, embedding2: Style embedding vectors

        Returns:
            Cosine similarity score in [0, 1]
        """
        similarity = np.dot(embedding1, embedding2) / (
            norm(embedding1) * norm(embedding2) + 1e-7
        )
        return (similarity + 1) / 2  # Map from [-1,1] to [0,1]

    def save_model(self, path='saved_models/style_encoder'):
        """Save encoder weights."""
        self.encoder.save_weights(path)

    def load_model(self, path='saved_models/style_encoder'):
        """Load pretrained encoder weights."""
        self.encoder.load_weights(path)

    def get_encoder(self):
        """Return the encoder model for inference."""
        return self.encoder

    def get_training_model(self):
        """Return the triplet model for training."""
        return self.training_model
