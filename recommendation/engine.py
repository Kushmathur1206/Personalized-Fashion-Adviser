"""
Advanced Recommendation Engine with LSTM-based Sequential Modeling.

Combines:
1. Multi-modal similarity (visual + style + category)
2. LSTM sequence model for user preference prediction
3. Weighted scoring with configurable importance
4. Nearest neighbor search with multiple distance metrics

The LSTM component models sequences of user interactions to predict
what they'll want next — not just what looks similar, but what fits
their evolving taste trajectory.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Embedding,
    Bidirectional, BatchNormalization, Masking
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


class SequencePredictor:
    """
    LSTM-based model that predicts the next preferred item embedding
    based on a sequence of previously interacted items.

    Input: Sequence of item embeddings (what user browsed/liked in order)
    Output: Predicted embedding of the next item user would prefer

    This captures temporal patterns like:
    - Users who browse formal shirts often look at ties next
    - Seasonal browsing patterns (winter coat -> scarf -> boots)
    - Style coherence across a shopping session
    """

    def __init__(self, embedding_dim=512, max_seq_length=20, lstm_units=256):
        """
        Args:
            embedding_dim: Dimension of item embeddings
            max_seq_length: Maximum number of past interactions to consider
            lstm_units: Number of LSTM hidden units
        """
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.lstm_units = lstm_units
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the LSTM sequence prediction model.

        Architecture:
        - Masking layer (handles variable-length sequences)
        - 2-layer Bidirectional LSTM (captures past and future context)
        - Dense layers map to predicted next-item embedding
        """
        input_seq = Input(
            shape=(self.max_seq_length, self.embedding_dim),
            name='interaction_sequence'
        )

        # Mask zero-padded positions (for shorter sequences)
        x = Masking(mask_value=0.0, name='mask_padding')(input_seq)

        # Bidirectional LSTM captures patterns in both directions
        x = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True,
                 dropout=0.3, recurrent_dropout=0.1),
            name='bilstm_1'
        )(x)
        x = Bidirectional(
            LSTM(self.lstm_units // 2, return_sequences=False,
                 dropout=0.3, recurrent_dropout=0.1),
            name='bilstm_2'
        )(x)

        # Predict next item embedding
        x = Dense(512, activation='relu', name='pred_dense1')(x)
        x = BatchNormalization(name='pred_bn1')(x)
        x = Dropout(0.3, name='pred_dropout1')(x)
        x = Dense(self.embedding_dim, activation='linear', name='predicted_embedding')(x)

        model = Model(inputs=input_seq, outputs=x, name='sequence_predictor')
        return model

    def compile_model(self, learning_rate=1e-3):
        """Compile with MSE loss (predict next embedding)."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['cosine_similarity']
        )

    def predict_next(self, interaction_history):
        """
        Predict the next preferred item embedding given interaction history.

        Args:
            interaction_history: List of embeddings (most recent interactions)
                                 Shape: (n_interactions, embedding_dim)

        Returns:
            Predicted next-item embedding of shape (embedding_dim,)
        """
        # Pad or truncate to max_seq_length
        seq = np.zeros((self.max_seq_length, self.embedding_dim))
        history = np.array(interaction_history)

        if len(history) > self.max_seq_length:
            history = history[-self.max_seq_length:]

        seq[-len(history):] = history  # Right-align (most recent at end)
        seq = np.expand_dims(seq, axis=0)

        predicted = self.model.predict(seq, verbose=0).flatten()
        return predicted

    def train(self, sequences, next_items, epochs=30, batch_size=32):
        """
        Train the sequence predictor.

        Args:
            sequences: Array of shape (n_samples, max_seq_length, embedding_dim)
            next_items: Array of shape (n_samples, embedding_dim) - ground truth next items
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        history = self.model.fit(
            sequences, next_items,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=3
                )
            ],
            verbose=1
        )
        return history

    def save_model(self, path='saved_models/sequence_predictor'):
        """Save model weights."""
        self.model.save_weights(path)

    def load_model(self, path='saved_models/sequence_predictor'):
        """Load pretrained weights."""
        self.model.load_weights(path)


class RecommendationEngine:
    """
    Multi-modal recommendation engine combining:
    1. Visual similarity (feature embeddings)
    2. Style similarity (triplet-loss embeddings)
    3. Category awareness (classifier predictions)
    4. LSTM sequence prediction (user behavior modeling)
    5. VAE-refined embeddings (denoised representations)

    Supports configurable weights for each modality and
    produces ranked, diversified recommendations.
    """

    def __init__(
        self,
        visual_weight=0.35,
        style_weight=0.25,
        category_weight=0.15,
        sequence_weight=0.25,
        n_neighbors=20,
        metric='cosine'
    ):
        """
        Args:
            visual_weight: Importance of visual similarity
            style_weight: Importance of style similarity
            category_weight: Importance of category match
            sequence_weight: Importance of LSTM sequence prediction
            n_neighbors: Number of candidates to retrieve
            metric: Distance metric for nearest neighbor search
        """
        self.visual_weight = visual_weight
        self.style_weight = style_weight
        self.category_weight = category_weight
        self.sequence_weight = sequence_weight
        self.n_neighbors = n_neighbors
        self.metric = metric

        # These get populated when index is built
        self.visual_index = None
        self.style_index = None
        self.refined_index = None
        self.visual_features = None
        self.style_features = None
        self.refined_features = None
        self.categories = None
        self.filenames = None

    def build_index(
        self,
        visual_features,
        style_features=None,
        refined_features=None,
        categories=None,
        filenames=None
    ):
        """
        Build the search indices for all modalities.

        Args:
            visual_features: Array of shape (n_items, visual_dim)
            style_features: Array of shape (n_items, style_dim) or None
            refined_features: Array of shape (n_items, refined_dim) or None
            categories: List of category labels per item or None
            filenames: List of file paths for each item
        """
        self.visual_features = visual_features
        self.style_features = style_features
        self.refined_features = refined_features
        self.categories = categories
        self.filenames = filenames

        # Build visual nearest neighbor index
        self.visual_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='brute',
            metric=self.metric
        )
        self.visual_index.fit(visual_features)

        # Build style index if available
        if style_features is not None:
            self.style_index = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm='brute',
                metric=self.metric
            )
            self.style_index.fit(style_features)

        # Build refined embedding index if available
        if refined_features is not None:
            self.refined_index = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm='brute',
                metric=self.metric
            )
            self.refined_index.fit(refined_features)

    def recommend(
        self,
        query_visual,
        query_style=None,
        query_category=None,
        interaction_history=None,
        sequence_predictor=None,
        n_results=10,
        category_filter=None
    ):
        """
        Generate recommendations using multi-modal weighted scoring.

        Args:
            query_visual: Visual embedding of query image
            query_style: Style embedding of query image (optional)
            query_category: Predicted category of query (optional)
            interaction_history: List of past interaction embeddings for LSTM (optional)
            sequence_predictor: Trained SequencePredictor instance (optional)
            n_results: Number of final recommendations
            category_filter: Only return items of this category (optional)

        Returns:
            Dictionary with:
            - indices: Indices of recommended items
            - scores: Combined similarity scores
            - distances: Per-modality distances
            - filenames: Paths of recommended items
        """
        candidate_scores = {}

        # 1. Visual similarity candidates
        if self.visual_index is not None:
            vis_distances, vis_indices = self.visual_index.kneighbors(
                [query_visual]
            )
            for idx, dist in zip(vis_indices[0], vis_distances[0]):
                similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                candidate_scores.setdefault(idx, {})
                candidate_scores[idx]['visual'] = similarity

        # 2. Style similarity candidates
        if query_style is not None and self.style_index is not None:
            style_distances, style_indices = self.style_index.kneighbors(
                [query_style]
            )
            for idx, dist in zip(style_indices[0], style_distances[0]):
                similarity = 1.0 / (1.0 + dist)
                candidate_scores.setdefault(idx, {})
                candidate_scores[idx]['style'] = similarity

        # 3. LSTM sequence prediction candidates
        if (interaction_history is not None and
                sequence_predictor is not None and
                len(interaction_history) > 0):
            predicted_embedding = sequence_predictor.predict_next(
                interaction_history
            )
            # Find items closest to predicted next embedding
            if self.visual_index is not None:
                seq_distances, seq_indices = self.visual_index.kneighbors(
                    [predicted_embedding]
                )
                for idx, dist in zip(seq_indices[0], seq_distances[0]):
                    similarity = 1.0 / (1.0 + dist)
                    candidate_scores.setdefault(idx, {})
                    candidate_scores[idx]['sequence'] = similarity

        # 4. Compute combined scores
        scored_candidates = []
        for idx, modality_scores in candidate_scores.items():
            # Category bonus/penalty
            category_bonus = 0.0
            if query_category and self.categories is not None:
                if idx < len(self.categories):
                    if self.categories[idx] == query_category:
                        category_bonus = self.category_weight

            # Apply category filter
            if category_filter and self.categories is not None:
                if idx < len(self.categories):
                    if self.categories[idx] != category_filter:
                        continue

            # Weighted combination
            visual_score = modality_scores.get('visual', 0.0) * self.visual_weight
            style_score = modality_scores.get('style', 0.0) * self.style_weight
            sequence_score = modality_scores.get('sequence', 0.0) * self.sequence_weight

            combined_score = visual_score + style_score + sequence_score + category_bonus
            scored_candidates.append((idx, combined_score, modality_scores))

        # Sort by combined score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top n_results
        top_candidates = scored_candidates[:n_results]

        indices = [c[0] for c in top_candidates]
        scores = [c[1] for c in top_candidates]
        details = [c[2] for c in top_candidates]

        result_filenames = []
        if self.filenames is not None:
            result_filenames = [
                self.filenames[i] for i in indices if i < len(self.filenames)
            ]

        return {
            'indices': indices,
            'scores': scores,
            'modality_details': details,
            'filenames': result_filenames
        }

    def recommend_simple(self, query_features, n_results=6):
        """
        Simple visual-only recommendation (backward compatible).
        Falls back to basic KNN if advanced features aren't available.

        Args:
            query_features: Visual embedding of query image
            n_results: Number of recommendations

        Returns:
            Indices of recommended items
        """
        if self.visual_index is None:
            # Fallback: brute force
            nn = NearestNeighbors(
                n_neighbors=n_results,
                algorithm='brute',
                metric='euclidean'
            )
            nn.fit(self.visual_features)
            distances, indices = nn.kneighbors([query_features])
            return indices[0]

        distances, indices = self.visual_index.kneighbors([query_features])
        return indices[0][:n_results]

    def get_similar_by_style(self, query_style, n_results=10):
        """Get items with similar style regardless of visual appearance."""
        if self.style_index is None:
            return []
        distances, indices = self.style_index.kneighbors([query_style])
        return indices[0][:n_results]

    def get_complementary(self, query_category, query_style, n_results=5):
        """
        Get complementary items (different category, similar style).
        E.g., if user uploads a formal shirt, suggest formal pants.

        Args:
            query_category: Category of the query item
            query_style: Style embedding of the query

        Returns:
            Indices of complementary items
        """
        if self.style_index is None or self.categories is None:
            return []

        # Find style-similar items
        distances, indices = self.style_index.kneighbors(
            [query_style], n_neighbors=self.n_neighbors * 3
        )

        # Filter to different category
        complementary = []
        for idx in indices[0]:
            if idx < len(self.categories):
                if self.categories[idx] != query_category:
                    complementary.append(idx)
                    if len(complementary) >= n_results:
                        break

        return complementary
