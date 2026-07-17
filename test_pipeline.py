"""Quick test: load data, build an outfit, verify it works."""
import pickle
import numpy as np
from outfit_builder import build_outfit, classify_upload

# Load catalog
embeddings = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

print(f"Catalog: {embeddings.shape[0]} items, {embeddings.shape[1]}-d embeddings")
print(f"Sample: {filenames[0]} -> {labels[0]}")

# Test outfit building for each occasion
test_idx = 0
test_label = labels[test_idx]
test_embedding = embeddings[test_idx]

print(f"\nUploaded: {test_label} (group: {classify_upload(test_label)})")

for occasion in ['Casual', 'Formal', 'Sporty', 'Party']:
    outfit = build_outfit(
        query_embedding=test_embedding,
        uploaded_label=test_label,
        occasion=occasion,
        catalog_embeddings=embeddings,
        catalog_labels=labels,
        catalog_filenames=filenames,
        n_options=3,
        uploaded_index=test_idx,
    )
    print(f"\n  {occasion}:")
    for slot, items in outfit.items():
        indices = [x['index'] for x in items]
        item_labels = [x['label'] for x in items]
        scores = [f"{x['score']:.2f}" for x in items]
        print(f"    {slot}: indices={indices} labels={item_labels} scores={scores}")
        # Verify all indices are unique
        assert len(set(indices)) == len(indices), "Duplicate indices!"

print("\nAll tests passed — indices are unique, items are different images.")
