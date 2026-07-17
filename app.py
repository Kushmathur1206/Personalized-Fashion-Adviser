"""
Embedding Generation Pipeline.

Processes all images in the catalog and generates:
1. Visual embeddings (ResNet50 feature extractor)
2. Style embeddings (triplet-loss style encoder)
3. Refined embeddings (LSTM-VAE autoencoder)
4. Category/attribute metadata (fashion classifier)

All outputs are saved as pickle files for use by the recommendation engine.

Usage:
    python app.py --images_dir images --backbone resnet50
    python app.py --images_dir images --backbone dual --batch_size 16
"""

import os
import argparse
import pickle

import numpy as np
from tqdm import tqdm

from models.feature_extractor import FashionFeatureExtractor
from models.fashion_classifier import FashionClassifier
from models.style_encoder import StyleEncoder
from models.autoencoder import FashionAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate fashion embeddings for the product catalog'
    )
    parser.add_argument(
        '--images_dir', type=str, default='images',
        help='Directory containing product images'
    )
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Directory to save embedding pickle files'
    )
    parser.add_argument(
        '--backbone', type=str, default='resnet50',
        choices=['resnet50', 'efficientnet', 'dual'],
        help='Feature extractor backbone architecture'
    )
    parser.add_argument(
        '--embedding_dim', type=int, default=512,
        help='Dimension of visual embeddings'
    )
    parser.add_argument(
        '--style_dim', type=int, default=256,
        help='Dimension of style embeddings'
    )
    parser.add_argument(
        '--latent_dim', type=int, default=128,
        help='Dimension of VAE-refined embeddings'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for processing images'
    )
    parser.add_argument(
        '--train_vae', action='store_true',
        help='Train the LSTM-VAE on extracted embeddings'
    )
    parser.add_argument(
        '--vae_epochs', type=int, default=50,
        help='Number of epochs for VAE training'
    )
    parser.add_argument(
        '--skip_style', action='store_true',
        help='Skip style embedding extraction'
    )
    parser.add_argument(
        '--skip_classifier', action='store_true',
        help='Skip category/attribute classification'
    )
    parser.add_argument(
        '--max_images', type=int, default=None,
        help='Limit number of images to process (for quick testing on CPU)'
    )
    return parser.parse_args()


def get_image_files(images_dir):
    """Collect all valid image files from the directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    filenames = []

    for file in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            filenames.append(os.path.join(images_dir, file))

    print(f"Found {len(filenames)} images in '{images_dir}'")
    return filenames


def extract_visual_embeddings(filenames, backbone, embedding_dim, batch_size):
    """Extract visual feature embeddings using the enhanced feature extractor."""
    print("\n" + "=" * 60)
    print("STEP 1: Extracting Visual Embeddings")
    print(f"  Backbone: {backbone}")
    print(f"  Embedding dim: {embedding_dim}")
    print("=" * 60)

    extractor = FashionFeatureExtractor(
        embedding_dim=embedding_dim,
        backbone=backbone,
        fine_tune_layers=20
    )

    print("Processing images in batches...")
    features = extractor.extract_features_batch(filenames, batch_size=batch_size)
    print(f"Visual embeddings shape: {features.shape}")

    return features, extractor


def extract_style_embeddings(filenames, style_dim, batch_size):
    """Extract style-aware embeddings using the triplet-loss encoder."""
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Style Embeddings")
    print(f"  Style dim: {style_dim}")
    print("=" * 60)

    style_encoder = StyleEncoder(embedding_dim=style_dim)

    print("Processing style embeddings...")
    style_features = style_encoder.encode_style_batch(
        filenames, batch_size=batch_size
    )
    print(f"Style embeddings shape: {style_features.shape}")

    return style_features, style_encoder


def classify_attributes(filenames, batch_size):
    """Run multi-task classification for category and attributes."""
    print("\n" + "=" * 60)
    print("STEP 3: Classifying Fashion Attributes")
    print("  Predicting: category, style, season, color")
    print("=" * 60)

    classifier = FashionClassifier()

    print("Classifying images...")
    metadata = classifier.predict_batch(filenames, batch_size=batch_size)
    print(f"Classified {len(metadata)} items")

    # Summary statistics
    categories = [m['category'] for m in metadata]
    unique_cats = set(categories)
    print(f"  Categories found: {len(unique_cats)}")
    for cat in sorted(unique_cats):
        count = categories.count(cat)
        print(f"    {cat}: {count} items")

    return metadata, classifier


def refine_with_vae(visual_features, latent_dim, train_vae, vae_epochs):
    """Refine embeddings using the LSTM-VAE."""
    print("\n" + "=" * 60)
    print("STEP 4: Refining Embeddings with LSTM-VAE")
    print(f"  Input dim: {visual_features.shape[1]}")
    print(f"  Latent dim: {latent_dim}")
    print("=" * 60)

    input_dim = visual_features.shape[1]
    # seq_length must divide input_dim evenly
    seq_length = 32
    while input_dim % seq_length != 0:
        seq_length -= 1

    autoencoder = FashionAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        seq_length=seq_length,
        beta=1.0
    )

    if train_vae:
        print(f"Training LSTM-VAE for {vae_epochs} epochs...")
        autoencoder.compile_model(learning_rate=1e-3)
        history = autoencoder.train(
            visual_features,
            epochs=vae_epochs,
            batch_size=64,
            validation_split=0.1
        )
        final_loss = history.history['loss'][-1]
        print(f"  Final training loss: {final_loss:.4f}")

        # Save trained VAE
        os.makedirs('saved_models', exist_ok=True)
        autoencoder.save_model('saved_models/autoencoder')
        print("  LSTM-VAE model saved to saved_models/")
    else:
        print("Skipping VAE training (use --train_vae to enable)")
        print("Using untrained VAE for dimensionality reduction...")

    # Generate refined embeddings
    refined_features = autoencoder.refine_embeddings_batch(visual_features)
    print(f"Refined embeddings shape: {refined_features.shape}")

    return refined_features, autoencoder


def save_outputs(output_dir, filenames, visual_features, style_features,
                 refined_features, metadata):
    """Save all generated data as pickle files."""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Visual embeddings (backward compatible with original format)
    embeddings_path = os.path.join(output_dir, 'embeddings.pkl')
    pickle.dump(list(visual_features), open(embeddings_path, 'wb'))
    print(f"  Visual embeddings -> {embeddings_path}")

    # Filenames
    filenames_path = os.path.join(output_dir, 'filenames.pkl')
    pickle.dump(filenames, open(filenames_path, 'wb'))
    print(f"  Filenames -> {filenames_path}")

    # Style embeddings
    if style_features is not None:
        style_path = os.path.join(output_dir, 'style_embeddings.pkl')
        pickle.dump(list(style_features), open(style_path, 'wb'))
        print(f"  Style embeddings -> {style_path}")

    # Refined (VAE) embeddings
    if refined_features is not None:
        refined_path = os.path.join(output_dir, 'refined_embeddings.pkl')
        pickle.dump(list(refined_features), open(refined_path, 'wb'))
        print(f"  Refined embeddings -> {refined_path}")

    # Item metadata (categories, styles, seasons, colors)
    if metadata is not None:
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        pickle.dump(metadata, open(metadata_path, 'wb'))
        print(f"  Item metadata -> {metadata_path}")

        # Also save categories as a simple list for backward compatibility
        categories = [m['category'] for m in metadata]
        categories_path = os.path.join(output_dir, 'categories.pkl')
        pickle.dump(categories, open(categories_path, 'wb'))
        print(f"  Categories -> {categories_path}")

    print(f"\n  Total items processed: {len(filenames)}")
    print("  All outputs saved successfully!")


def main():
    args = parse_args()

    # Validate input directory
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' not found.")
        print("Please ensure your product images are in the specified directory.")
        return

    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Collect image files
    filenames = get_image_files(args.images_dir)
    if not filenames:
        print("No valid image files found. Supported formats: jpg, jpeg, png, bmp, webp")
        return

    # Limit number of images if specified (useful for CPU testing)
    if args.max_images and args.max_images < len(filenames):
        filenames = filenames[:args.max_images]
        print(f"Limited to {args.max_images} images (use --max_images to adjust)")

    # Step 1: Visual embeddings
    visual_features, _ = extract_visual_embeddings(
        filenames, args.backbone, args.embedding_dim, args.batch_size
    )

    # Step 2: Style embeddings
    style_features = None
    if not args.skip_style:
        style_features, _ = extract_style_embeddings(
            filenames, args.style_dim, args.batch_size
        )

    # Step 3: Category/attribute classification
    metadata = None
    if not args.skip_classifier:
        metadata, _ = classify_attributes(filenames, args.batch_size)

    # Step 4: LSTM-VAE refinement
    refined_features, _ = refine_with_vae(
        visual_features, args.latent_dim, args.train_vae, args.vae_epochs
    )

    # Save everything
    save_outputs(
        args.output_dir, filenames, visual_features,
        style_features, refined_features, metadata
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Generated embeddings for {len(filenames)} products")
    print(f"Run 'streamlit run main.py' to start the recommendation app")


if __name__ == '__main__':
    main()
