"""
Outfit Builder — core logic.

Responsibilities:
1. Define outfit slots (what categories make an outfit)
2. Define occasion rules (what's appropriate for each context)
3. Fill each slot using occasion-aware scoring (not just visual similarity)

Key insight: occasion changes HOW items are ranked, not just which categories
are eligible. We use "occasion anchors" — for each occasion, we compute an
ideal embedding direction by averaging items that belong to that occasion's
typical categories. This pulls results toward the right vibe.
"""

import numpy as np
from numpy.linalg import norm


# ─────────────────────────────────────────────────────────────
# Category Mapping
# ─────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    'T-Shirt': 'top',
    'Shirt': 'top',
    'Longsleeve': 'top',
    'Polo': 'top',
    'Hoodie': 'top',
    'Blouse': 'top',
    'Top': 'top',
    'Undershirt': 'top',
    'Blazer': 'outerwear',
    'Outwear': 'outerwear',
    'Pants': 'bottom',
    'Shorts': 'bottom',
    'Skirt': 'bottom',
    'Dress': 'dress',
    'Shoes': 'shoes',
    'Hat': 'accessory',
    'Body': 'dress',
}

# ─────────────────────────────────────────────────────────────
# Outfit Slot Rules
# ─────────────────────────────────────────────────────────────

OUTFIT_SLOTS = {
    'top': ['bottom', 'shoes'],
    'bottom': ['top', 'shoes'],
    'dress': ['shoes', 'outerwear'],
    'shoes': ['top', 'bottom'],
    'outerwear': ['top', 'bottom', 'shoes'],
    'accessory': ['top', 'bottom', 'shoes'],
}

# ─────────────────────────────────────────────────────────────
# Occasion Configuration
# ─────────────────────────────────────────────────────────────

# allowed_labels: which item labels are eligible for this slot
# anchor_labels: which labels define the "vibe" for this occasion
#   (used to compute an occasion embedding direction)
OCCASIONS = {
    'Casual': {
        'top': {
            'allowed': ['T-Shirt', 'Polo', 'Hoodie', 'Longsleeve'],
            'anchor': ['T-Shirt', 'Polo', 'Hoodie', 'Shorts'],
        },
        'bottom': {
            'allowed': ['Pants', 'Shorts'],
            'anchor': ['Shorts', 'T-Shirt', 'Hoodie'],
        },
        'shoes': {
            'allowed': ['Shoes'],
            'anchor': ['Shorts', 'T-Shirt', 'Hoodie'],
        },
        'outerwear': {
            'allowed': ['Hoodie', 'Outwear'],
            'anchor': ['T-Shirt', 'Shorts'],
        },
        'accessory': {
            'allowed': ['Hat'],
            'anchor': ['T-Shirt', 'Shorts'],
        },
    },
    'Formal': {
        'top': {
            'allowed': ['Shirt', 'Blouse', 'Longsleeve'],
            'anchor': ['Shirt', 'Blouse', 'Blazer', 'Pants'],
        },
        'bottom': {
            'allowed': ['Pants', 'Skirt'],
            'anchor': ['Shirt', 'Blazer', 'Blouse'],
        },
        'shoes': {
            'allowed': ['Shoes'],
            'anchor': ['Shirt', 'Blazer', 'Pants', 'Skirt'],
        },
        'outerwear': {
            'allowed': ['Blazer'],
            'anchor': ['Shirt', 'Pants'],
        },
        'accessory': {
            'allowed': ['Hat'],
            'anchor': ['Shirt', 'Blazer'],
        },
    },
    'Sporty': {
        'top': {
            'allowed': ['T-Shirt', 'Polo', 'Hoodie'],
            'anchor': ['Hoodie', 'Shorts', 'T-Shirt'],
        },
        'bottom': {
            'allowed': ['Shorts', 'Pants'],
            'anchor': ['Hoodie', 'T-Shirt', 'Polo'],
        },
        'shoes': {
            'allowed': ['Shoes'],
            'anchor': ['Hoodie', 'Shorts', 'T-Shirt', 'Polo'],
        },
        'outerwear': {
            'allowed': ['Hoodie', 'Outwear'],
            'anchor': ['Shorts', 'T-Shirt'],
        },
        'accessory': {
            'allowed': ['Hat'],
            'anchor': ['Hoodie', 'Shorts'],
        },
    },
    'Party': {
        'top': {
            'allowed': ['Shirt', 'Blouse', 'Top'],
            'anchor': ['Dress', 'Skirt', 'Blouse', 'Top'],
        },
        'bottom': {
            'allowed': ['Pants', 'Skirt'],
            'anchor': ['Dress', 'Blouse', 'Top', 'Shirt'],
        },
        'shoes': {
            'allowed': ['Shoes'],
            'anchor': ['Dress', 'Skirt', 'Blouse', 'Top'],
        },
        'outerwear': {
            'allowed': ['Blazer'],
            'anchor': ['Dress', 'Shirt'],
        },
        'accessory': {
            'allowed': ['Hat'],
            'anchor': ['Dress', 'Top'],
        },
    },
}


def classify_upload(label: str) -> str:
    """Map a dataset label to an outfit group."""
    return CATEGORY_MAP.get(label, 'top')


def get_slots_to_fill(uploaded_group: str) -> list:
    """Given uploaded item's group, return which slots need filling."""
    return OUTFIT_SLOTS.get(uploaded_group, ['top', 'bottom', 'shoes'])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    d = norm(a) * norm(b)
    if d == 0:
        return 0.0
    return float(np.dot(a, b) / d)


def compute_occasion_anchor(
    anchor_labels: list,
    catalog_embeddings: np.ndarray,
    catalog_labels: list,
) -> np.ndarray:
    """
    Compute an "occasion direction" embedding by averaging embeddings
    of items that typify this occasion.

    This gives us a vector that points toward the occasion's vibe.
    """
    anchor_set = set(anchor_labels)
    anchor_embeddings = []

    for i, label in enumerate(catalog_labels):
        if label in anchor_set:
            anchor_embeddings.append(catalog_embeddings[i])

    if not anchor_embeddings:
        return None

    anchor = np.mean(anchor_embeddings, axis=0)
    n = norm(anchor)
    if n > 0:
        anchor = anchor / n
    return anchor


def fill_slot(
    slot: str,
    occasion: str,
    query_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_labels: list,
    catalog_filenames: list,
    n_options: int = 3,
    exclude_indices: set = None,
) -> list:
    """
    Find the best items for a given outfit slot.

    Scoring combines:
    - Style similarity to the uploaded item (40%)
    - Similarity to the occasion anchor (60%)

    This means: same uploaded item + different occasion = different results
    because the occasion anchor pulls the ranking in a different direction.
    """
    if exclude_indices is None:
        exclude_indices = set()

    # Get occasion config for this slot
    occasion_config = OCCASIONS.get(occasion, OCCASIONS['Casual'])
    slot_config = occasion_config.get(slot, {'allowed': [], 'anchor': []})
    allowed_labels = set(slot_config['allowed'])
    anchor_labels = slot_config['anchor']

    # Filter catalog to eligible items
    candidates = []
    for i, label in enumerate(catalog_labels):
        if i in exclude_indices:
            continue
        if label in allowed_labels:
            candidates.append(i)

    # Fallback: any item in this slot group
    if not candidates:
        for i, label in enumerate(catalog_labels):
            if i in exclude_indices:
                continue
            if CATEGORY_MAP.get(label) == slot:
                candidates.append(i)

    if not candidates:
        return []

    # Compute occasion anchor
    occasion_anchor = compute_occasion_anchor(
        anchor_labels, catalog_embeddings, catalog_labels
    )

    # Score each candidate
    scores = []
    for idx in candidates:
        item_emb = catalog_embeddings[idx]

        # Style match to uploaded item
        style_score = cosine_similarity(query_embedding, item_emb)

        # Occasion match (how well does this item fit the occasion vibe)
        if occasion_anchor is not None:
            occasion_score = cosine_similarity(occasion_anchor, item_emb)
        else:
            occasion_score = 0.0

        # Combined score: occasion dominates so different occasions give different results
        combined = 0.4 * style_score + 0.6 * occasion_score
        scores.append((idx, combined))

    # Sort by combined score
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top N with diversity filter
    results = []
    selected_embeddings = []

    for idx, score in scores:
        if len(results) >= n_options:
            break

        emb = catalog_embeddings[idx]
        too_similar = False
        for sel_emb in selected_embeddings:
            if cosine_similarity(emb, sel_emb) > 0.85:
                too_similar = True
                break

        if not too_similar:
            results.append({
                'index': idx,
                'filename': catalog_filenames[idx],
                'label': catalog_labels[idx],
                'score': score,
            })
            selected_embeddings.append(emb)

    return results


def build_outfit(
    query_embedding: np.ndarray,
    uploaded_label: str,
    occasion: str,
    catalog_embeddings: np.ndarray,
    catalog_labels: list,
    catalog_filenames: list,
    n_options: int = 3,
    uploaded_index: int = None,
) -> dict:
    """
    Build a complete outfit given an uploaded item and occasion.

    Returns:
        Dict with slot names as keys, each containing list of options
    """
    uploaded_group = classify_upload(uploaded_label)
    slots_to_fill = get_slots_to_fill(uploaded_group)

    exclude = {uploaded_index} if uploaded_index is not None else set()

    outfit = {}
    for slot in slots_to_fill:
        options = fill_slot(
            slot=slot,
            occasion=occasion,
            query_embedding=query_embedding,
            catalog_embeddings=catalog_embeddings,
            catalog_labels=catalog_labels,
            catalog_filenames=catalog_filenames,
            n_options=n_options,
            exclude_indices=exclude,
        )
        if options:
            outfit[slot] = options

    return outfit
