"""
Fashion Category & Attribute Classifier.

Multi-task classification network that predicts:
1. Product category (tops, bottoms, shoes, accessories, etc.)
2. Style attributes (casual, formal, sporty, etc.)
3. Color palette (dominant colors)
4. Season suitability (summer, winter, etc.)

This enables category-aware recommendations and attribute-based filtering.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam


# Fashion category labels (based on common fashion datasets)
CATEGORY_LABELS = [
    'Topwear', 'Bottomwear', 'Shoes', 'Bags', 'Accessories',
    'Innerwear', 'Dress', 'Loungewear', 'Sandals', 'Watches',
    'Belts', 'Flip Flops', 'Headwear', 'Saree', 'Jewellery',
    'Sunglasses', 'Wallets', 'Fragrance', 'Ties', 'Scarves'
]

STYLE_LABELS = [
    'Casual', 'Formal', 'Sports', 'Ethnic', 'Party',
    'Smart Casual', 'Travel', 'Streetwear'
]

SEASON_LABELS = ['Spring', 'Summer', 'Autumn', 'Winter', 'All-Season']

COLOR_LABELS = [
    'Black', 'White', 'Red', 'Blue', 'Green', 'Yellow',
    'Pink', 'Purple', 'Orange', 'Brown', 'Grey', 'Beige',
    'Navy', 'Maroon', 'Multi-color'
]


class FashionClassifier:
    """
    Multi-task fashion classifier that predicts category, style,
    season, and color attributes simultaneously.

    Uses shared backbone with task-specific heads for efficient
    multi-attribute prediction.
    """

    def __init__(self, num_categories=20, num_styles=8, num_seasons=5, num_colors=15):
        """
        Args:
            num_categories: Number of fashion categories
            num_styles: Number of style types
            num_seasons: Number of season labels
            num_colors: Number of color labels
        """
        self.num_categories = num_categories
        self.num_styles = num_styles
        self.num_seasons = num_seasons
        self.num_colors = num_colors
        self.model = self._build_model()

    def _build_model(self):
        """
        Build multi-task classification model with shared backbone
        and separate heads for each attribute.
        """
        input_tensor = Input(shape=(224, 224, 3))

        # Shared backbone
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        # Freeze most layers, fine-tune top 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Shared feature layer
        shared = Dense(1024, activation='relu', name='shared_dense')(x)
        shared = BatchNormalization(name='shared_bn')(shared)
        shared = Dropout(0.4, name='shared_dropout')(shared)

        # Category classification head
        cat_x = Dense(256, activation='relu', name='cat_dense')(shared)
        cat_x = Dropout(0.3, name='cat_dropout')(cat_x)
        category_output = Dense(
            self.num_categories, activation='softmax', name='category_output'
        )(cat_x)

        # Style classification head (multi-label)
        style_x = Dense(256, activation='relu', name='style_dense')(shared)
        style_x = Dropout(0.3, name='style_dropout')(style_x)
        style_output = Dense(
            self.num_styles, activation='sigmoid', name='style_output'
        )(style_x)

        # Season classification head (multi-label)
        season_x = Dense(128, activation='relu', name='season_dense')(shared)
        season_x = Dropout(0.3, name='season_dropout')(season_x)
        season_output = Dense(
            self.num_seasons, activation='sigmoid', name='season_output'
        )(season_x)

        # Color classification head (multi-label)
        color_x = Dense(128, activation='relu', name='color_dense')(shared)
        color_x = Dropout(0.3, name='color_dropout')(color_x)
        color_output = Dense(
            self.num_colors, activation='sigmoid', name='color_output'
        )(color_x)

        model = Model(
            inputs=input_tensor,
            outputs=[category_output, style_output, season_output, color_output],
            name='fashion_classifier'
        )

        return model

    def compile_model(self, learning_rate=1e-4):
        """Compile with multi-task loss."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'category_output': 'categorical_crossentropy',
                'style_output': 'binary_crossentropy',
                'season_output': 'binary_crossentropy',
                'color_output': 'binary_crossentropy'
            },
            loss_weights={
                'category_output': 1.0,
                'style_output': 0.5,
                'season_output': 0.3,
                'color_output': 0.3
            },
            metrics={
                'category_output': 'accuracy',
                'style_output': 'binary_accuracy',
                'season_output': 'binary_accuracy',
                'color_output': 'binary_accuracy'
            }
        )

    def predict(self, img_path):
        """
        Predict all attributes for a single image.

        Args:
            img_path: Path to the image file

        Returns:
            Dictionary with predicted category, styles, seasons, and colors
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        predictions = self.model.predict(preprocessed_img, verbose=0)
        cat_pred, style_pred, season_pred, color_pred = predictions

        # Decode predictions
        category_idx = np.argmax(cat_pred[0])
        category = CATEGORY_LABELS[category_idx] if category_idx < len(CATEGORY_LABELS) else 'Unknown'
        category_confidence = float(cat_pred[0][category_idx])

        # Multi-label: select attributes above threshold
        style_threshold = 0.4
        styles = [
            STYLE_LABELS[i] for i, v in enumerate(style_pred[0])
            if v > style_threshold and i < len(STYLE_LABELS)
        ]
        if not styles:
            styles = [STYLE_LABELS[np.argmax(style_pred[0])]]

        season_threshold = 0.4
        seasons = [
            SEASON_LABELS[i] for i, v in enumerate(season_pred[0])
            if v > season_threshold and i < len(SEASON_LABELS)
        ]
        if not seasons:
            seasons = ['All-Season']

        color_threshold = 0.3
        colors = [
            COLOR_LABELS[i] for i, v in enumerate(color_pred[0])
            if v > color_threshold and i < len(COLOR_LABELS)
        ]
        if not colors:
            colors = [COLOR_LABELS[np.argmax(color_pred[0])]]

        return {
            'category': category,
            'category_confidence': category_confidence,
            'styles': styles,
            'seasons': seasons,
            'colors': colors,
            'raw_category_probs': cat_pred[0],
            'raw_style_probs': style_pred[0],
            'raw_season_probs': season_pred[0],
            'raw_color_probs': color_pred[0]
        }

    def predict_batch(self, img_paths, batch_size=32):
        """Predict attributes for multiple images."""
        results = []
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = image.load_img(path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    batch_images.append(img_array)
                except Exception:
                    batch_images.append(np.zeros((224, 224, 3)))

            batch_array = np.array(batch_images)
            preprocessed_batch = preprocess_input(batch_array)
            predictions = self.model.predict(preprocessed_batch, verbose=0)

            cat_preds, style_preds, season_preds, color_preds = predictions

            for j in range(len(batch_paths)):
                category_idx = np.argmax(cat_preds[j])
                category = CATEGORY_LABELS[category_idx] if category_idx < len(CATEGORY_LABELS) else 'Unknown'

                styles = [
                    STYLE_LABELS[k] for k, v in enumerate(style_preds[j])
                    if v > 0.4 and k < len(STYLE_LABELS)
                ]
                if not styles:
                    styles = [STYLE_LABELS[np.argmax(style_preds[j])]]

                seasons = [
                    SEASON_LABELS[k] for k, v in enumerate(season_preds[j])
                    if v > 0.4 and k < len(SEASON_LABELS)
                ]
                if not seasons:
                    seasons = ['All-Season']

                colors = [
                    COLOR_LABELS[k] for k, v in enumerate(color_preds[j])
                    if v > 0.3 and k < len(COLOR_LABELS)
                ]
                if not colors:
                    colors = [COLOR_LABELS[np.argmax(color_preds[j])]]

                results.append({
                    'category': category,
                    'category_confidence': float(cat_preds[j][category_idx]),
                    'styles': styles,
                    'seasons': seasons,
                    'colors': colors
                })

        return results

    def save_model(self, path='saved_models/fashion_classifier'):
        """Save model weights."""
        self.model.save_weights(path)

    def load_model(self, path='saved_models/fashion_classifier'):
        """Load pretrained model weights."""
        self.model.load_weights(path)

    def get_model(self):
        """Return the underlying Keras model for training."""
        return self.model
