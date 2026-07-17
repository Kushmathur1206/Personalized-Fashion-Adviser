"""
Embedding Pipeline — generates embeddings + labels for the catalog.

Reads images and their labels from the clothing-dataset,
extracts ResNet50 embeddings, and saves:
- embeddings.pkl (normalized feature vectors)
- filenames.pkl (image paths)
- labels.pkl (category label per image)

Usage:
    python app.py --images_dir clothing-dataset/images --labels_csv clothing-dataset/images.csv
    python app.py --images_dir clothing-dataset/images --labels_csv clothing-dataset/images.csv --max_images 300
"""

import os
import csv
import argparse
import pickle

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image as keras_image
from numpy.linalg import norm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate catalog embeddings')
    parser.add_argument('--images_dir', type=str, default='clothing-dataset/images')
    parser.add_argument('--labels_csv', type=str, default='clothing-dataset/images.csv')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit images to process (for fast testing on CPU)')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def load_labels(csv_path):
    """Load image_id -> label mapping from CSV."""
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            if label in ('Not sure', 'Skip', 'Other'):
                continue
            labels[row['image']] = label
    return labels


def build_model():
    """ResNet50 + GlobalAveragePooling = 2048-d embedding."""
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = Sequential([base, GlobalAveragePooling2D()])
    return model


def extract_batch(paths, model, batch_size=32):
    """Extract normalized embeddings in batches."""
    all_features = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_images = []

        for p in batch_paths:
            try:
                img = keras_image.load_img(p, target_size=(224, 224))
                batch_images.append(keras_image.img_to_array(img))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
                batch_images.append(np.zeros((224, 224, 3)))

        batch_array = preprocess_input(np.array(batch_images))
        features = model.predict(batch_array, verbose=0)

        # Normalize each vector
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-7
        all_features.append(features / norms)

        done = min(i + batch_size, len(paths))
        print(f"  Processed {done}/{len(paths)} images", end='\r')

    print()
    return np.vstack(all_features)


def main():
    args = parse_args()

    # Load labels
    print("Loading labels...")
    label_map = load_labels(args.labels_csv)
    print(f"  {len(label_map)} labeled images (excluded 'Not sure', 'Skip', 'Other')")

    # Collect valid image files that have labels
    filenames = []
    labels = []

    for image_id, label in label_map.items():
        # Dataset uses UUID filenames with .jpg extension
        path = os.path.join(args.images_dir, f"{image_id}.jpg")
        if os.path.exists(path):
            filenames.append(path)
            labels.append(label)

    print(f"  {len(filenames)} images found on disk")

    if not filenames:
        print("Error: No images found. Check --images_dir and --labels_csv paths.")
        return

    # Limit if requested
    if args.max_images and args.max_images < len(filenames):
        filenames = filenames[:args.max_images]
        labels = labels[:args.max_images]
        print(f"  Limited to {args.max_images} images")

    # Extract embeddings
    print("\nBuilding model...")
    model = build_model()

    print(f"\nExtracting embeddings ({len(filenames)} images)...")
    embeddings = extract_batch(filenames, model, batch_size=args.batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save
    print("\nSaving...")
    pickle.dump(list(embeddings), open('embeddings.pkl', 'wb'))
    pickle.dump(filenames, open('filenames.pkl', 'wb'))
    pickle.dump(labels, open('labels.pkl', 'wb'))

    print(f"\n  embeddings.pkl  ({embeddings.shape[0]} x {embeddings.shape[1]})")
    print(f"  filenames.pkl   ({len(filenames)} paths)")
    print(f"  labels.pkl      ({len(labels)} labels)")

    # Print label distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"\n  Label distribution:")
    for label, count in dist.most_common():
        print(f"    {label}: {count}")

    # Train LSTM outfit model
    print("\n" + "=" * 50)
    print("Training LSTM Outfit Sequence Model...")
    print("=" * 50)

    from lstm_outfit import OutfitLSTM, generate_training_sequences

    # Generate synthetic outfit sequences from catalog
    print("  Generating training sequences...")
    X_train, Y_train = generate_training_sequences(embeddings, labels, n_sequences=3000)
    print(f"  Training data: {X_train.shape[0]} samples")

    # Train
    lstm_model = OutfitLSTM()
    lstm_model.compile(lr=1e-3)
    print("  Training LSTM...")
    lstm_model.train(X_train, Y_train, epochs=20, batch_size=64)

    # Save
    os.makedirs('saved_models', exist_ok=True)
    lstm_model.save('saved_models/outfit_lstm.weights.h5')
    print("  LSTM model saved to saved_models/outfit_lstm.weights.h5")

    print("\nDone. Run: streamlit run main.py")


if __name__ == '__main__':
    main()
