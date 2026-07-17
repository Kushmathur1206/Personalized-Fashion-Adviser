"""
LSTM Outfit Sequence Model.

Generates outfit items sequentially — each prediction is conditioned on
all previously selected items + the occasion.

Input per time step:
    [item_embedding (2048-d)] + [occasion_one_hot (4-d)] + [category_one_hot (7-d)]
    = 2059-d

The LSTM learns:
- "After a casual top, suggest a casual bottom"
- "After a formal shirt + formal pants, suggest formal shoes"
- Visual compatibility across the sequence (colors, textures)

Training: synthetic outfit sequences built from the catalog by pairing
items across categories that share style similarity.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Masking, Bidirectional, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from numpy.linalg import norm

from outfit_builder import CATEGORY_MAP, OCCASIONS


# ─────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────

OCCASION_LIST = ['Casual', 'Formal', 'Sporty', 'Party']
CATEGORY_GROUP_LIST = ['top', 'bottom', 'shoes', 'dress', 'outerwear', 'accessory', 'other']

EMBEDDING_DIM = 2048
OCCASION_DIM = len(OCCASION_LIST)        # 4
CATEGORY_DIM = len(CATEGORY_GROUP_LIST)  # 7
INPUT_DIM = EMBEDDING_DIM + OCCASION_DIM + CATEGORY_DIM  # 2059
MAX_SEQ_LEN = 4  # max items in an outfit sequence


def encode_occasion(occasion: str) -> np.ndarray:
    """One-hot encode occasion."""
    vec = np.zeros(OCCASION_DIM)
    if occasion in OCCASION_LIST:
        vec[OCCASION_LIST.index(occasion)] = 1.0
    return vec


def encode_category(label: str) -> np.ndarray:
    """One-hot encode category group."""
    vec = np.zeros(CATEGORY_DIM)
    group = CATEGORY_MAP.get(label, 'other')
    if group in CATEGORY_GROUP_LIST:
        vec[CATEGORY_GROUP_LIST.index(group)] = 1.0
    return vec


def build_input_vector(embedding: np.ndarray, occasion: str, label: str) -> np.ndarray:
    """Combine embedding + occasion + category into single input vector."""
    occ_vec = encode_occasion(occasion)
    cat_vec = encode_category(label)
    return np.concatenate([embedding, occ_vec, cat_vec])


# ─────────────────────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────────────────────

class OutfitLSTM:
    """
    LSTM that predicts the next outfit item embedding given
    the sequence of items selected so far.
    """

    def __init__(self):
        self.model = self._build()

    def _build(self):
        """Build the sequence model."""
        inp = Input(shape=(MAX_SEQ_LEN, INPUT_DIM), name='outfit_sequence')

        # Mask padding (zero vectors for empty slots)
        x = Masking(mask_value=0.0)(inp)

        # Bidirectional LSTM — captures context in both directions
        x = Bidirectional(LSTM(256, return_sequences=False, dropout=0.3), name='bilstm')(x)

        # Predict next item embedding
        x = Dense(512, activation='relu', name='dense1')(x)
        x = Dropout(0.2)(x)
        x = Dense(EMBEDDING_DIM, activation='linear', name='predicted_embedding')(x)

        model = Model(inputs=inp, outputs=x, name='outfit_lstm')
        return model

    def compile(self, lr=1e-3):
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    def predict_next(self, sequence_embeddings, sequence_labels, occasion):
        """
        Given items selected so far, predict the next item embedding.

        Args:
            sequence_embeddings: list of embeddings selected so far
            sequence_labels: list of labels for those items
            occasion: current occasion string

        Returns:
            Predicted embedding (2048-d) for the next item
        """
        # Build input sequence (right-aligned, zero-padded on left)
        seq = np.zeros((MAX_SEQ_LEN, INPUT_DIM))
        n = min(len(sequence_embeddings), MAX_SEQ_LEN)

        for i in range(n):
            idx = -(n - i)  # right-align
            vec = build_input_vector(
                sequence_embeddings[i], occasion, sequence_labels[i]
            )
            seq[MAX_SEQ_LEN + idx] = vec

        # Predict
        seq_batch = np.expand_dims(seq, axis=0)
        predicted = self.model.predict(seq_batch, verbose=0).flatten()
        return predicted

    def train(self, sequences_X, next_items_Y, epochs=30, batch_size=32):
        """
        Train on outfit sequences.

        Args:
            sequences_X: shape (n_samples, MAX_SEQ_LEN, INPUT_DIM)
            next_items_Y: shape (n_samples, EMBEDDING_DIM) — the target next embedding
        """
        history = self.model.fit(
            sequences_X, next_items_Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            ],
            verbose=1,
        )
        return history

    def save(self, path='saved_models/outfit_lstm.weights.h5'):
        self.model.save_weights(path)

    def load(self, path='saved_models/outfit_lstm.weights.h5'):
        self.model.load_weights(path)


# ─────────────────────────────────────────────────────────────
# Training Data Generation
# ─────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    d = norm(a) * norm(b)
    if d == 0:
        return 0.0
    return float(np.dot(a, b) / d)


def generate_training_sequences(
    embeddings: np.ndarray,
    labels: list,
    n_sequences: int = 2000,
) -> tuple:
    """
    Generate synthetic outfit sequences for LSTM training.

    Strategy: for each occasion, pick a random top → find a style-compatible
    bottom → find style-compatible shoes → that's one training sequence.

    For each sequence of length N, we create N-1 training samples:
      [top] → bottom  (predict bottom given top)
      [top, bottom] → shoes  (predict shoes given top+bottom)
    """
    # Group items by category group
    groups = {}
    for i, label in enumerate(labels):
        group = CATEGORY_MAP.get(label, 'other')
        if group not in groups:
            groups[group] = []
        groups[group].append(i)

    # Outfit templates: ordered sequence of slots
    templates = [
        ['top', 'bottom', 'shoes'],
        ['top', 'bottom', 'shoes', 'accessory'],
        ['dress', 'shoes'],
        ['top', 'bottom', 'shoes', 'outerwear'],
    ]

    sequences_X = []
    next_items_Y = []

    rng = np.random.default_rng(42)

    for _ in range(n_sequences):
        # Pick random occasion
        occasion = rng.choice(OCCASION_LIST)

        # Pick random template that has items available
        rng.shuffle(templates)
        template = None
        for t in templates:
            if all(slot in groups and len(groups[slot]) > 0 for slot in t):
                template = t
                break

        if template is None:
            continue

        # Build an outfit by picking style-compatible items
        outfit_indices = []
        outfit_labels_seq = []

        for slot_idx, slot in enumerate(template):
            pool = groups[slot]

            if slot_idx == 0:
                # First item: random
                chosen = rng.choice(pool)
            else:
                # Pick item most similar to average of outfit so far
                outfit_emb_avg = np.mean(
                    [embeddings[i] for i in outfit_indices], axis=0
                )
                # Score all items in pool
                scored = []
                for idx in pool:
                    sim = cosine_sim(outfit_emb_avg, embeddings[idx])
                    scored.append((idx, sim))
                scored.sort(key=lambda x: x[1], reverse=True)

                # Pick from top 5 with some randomness
                top_k = min(5, len(scored))
                pick = rng.integers(0, top_k)
                chosen = scored[pick][0]

            outfit_indices.append(chosen)
            outfit_labels_seq.append(labels[chosen])

        # Create training samples from this outfit
        for step in range(1, len(outfit_indices)):
            # Input: items 0..step-1
            seq = np.zeros((MAX_SEQ_LEN, INPUT_DIM))
            for i in range(step):
                vec = build_input_vector(
                    embeddings[outfit_indices[i]], occasion, outfit_labels_seq[i]
                )
                # Right-align
                pos = MAX_SEQ_LEN - step + i
                if pos >= 0:
                    seq[pos] = vec

            # Target: embedding of item at position 'step'
            target = embeddings[outfit_indices[step]]

            sequences_X.append(seq)
            next_items_Y.append(target)

    return np.array(sequences_X), np.array(next_items_Y)


# ─────────────────────────────────────────────────────────────
# Inference: generate full outfit with LSTM
# ─────────────────────────────────────────────────────────────

def generate_outfit_lstm(
    lstm_model: OutfitLSTM,
    query_embedding: np.ndarray,
    query_label: str,
    occasion: str,
    catalog_embeddings: np.ndarray,
    catalog_labels: list,
    catalog_filenames: list,
    max_items: int = 3,
) -> list:
    """
    Generate outfit sequentially using LSTM.

    Each item depends on all previously selected items.

    Returns:
        List of dicts: [{index, filename, label, step}, ...]
    """
    # Determine which slots to generate
    query_group = CATEGORY_MAP.get(query_label, 'top')
    if query_group == 'top':
        target_slots = ['bottom', 'shoes', 'accessory']
    elif query_group == 'bottom':
        target_slots = ['top', 'shoes', 'accessory']
    elif query_group == 'dress':
        target_slots = ['shoes', 'outerwear']
    else:
        target_slots = ['top', 'bottom', 'shoes']

    target_slots = target_slots[:max_items]

    # Start sequence with uploaded item
    current_embeddings = [query_embedding]
    current_labels = [query_label]
    used_indices = set()

    results = []

    for slot in target_slots:
        # Get eligible items for this slot
        eligible = []
        for i, label in enumerate(catalog_labels):
            if i in used_indices:
                continue
            if CATEGORY_MAP.get(label) == slot:
                eligible.append(i)

        if not eligible:
            continue

        # LSTM predicts next embedding
        predicted_emb = lstm_model.predict_next(
            current_embeddings, current_labels, occasion
        )

        # Find closest match among eligible items
        best_idx = None
        best_sim = -1.0

        for idx in eligible:
            sim = cosine_sim(predicted_emb, catalog_embeddings[idx])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx is not None:
            results.append({
                'index': best_idx,
                'filename': catalog_filenames[best_idx],
                'label': catalog_labels[best_idx],
                'score': best_sim,
                'slot': slot,
            })
            # Add to sequence for next prediction
            current_embeddings.append(catalog_embeddings[best_idx])
            current_labels.append(catalog_labels[best_idx])
            used_indices.add(best_idx)

    return results
