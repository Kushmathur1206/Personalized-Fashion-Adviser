"""
Test Script for the Deep Learning Fashion Recommendation Pipeline.

Tests each component of the system:
1. Feature Extractor (ResNet50 / EfficientNet / Dual backbone)
2. Fashion Classifier (multi-task category/style/season/color)
3. Style Encoder (triplet-loss embeddings)
4. LSTM-VAE Autoencoder (embedding refinement)
5. Recommendation Engine (multi-modal + LSTM sequence predictor)
6. Diversity Re-ranker (MMR)
7. Attribute Filters

Usage:
    python test.py                          # Run all tests with sample image
    python test.py --image path/to/img.jpg  # Test with a specific image
    python test.py --test_mode quick        # Skip slow model builds
"""

import os
import argparse
import time

import numpy as np

# Suppress TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(description='Test the fashion recommendation pipeline')
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to a test image (uses synthetic data if not provided)'
    )
    parser.add_argument(
        '--test_mode', type=str, default='full',
        choices=['full', 'quick'],
        help='full = test all models, quick = skip slow model instantiation'
    )
    return parser.parse_args()


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_pass(msg):
    print(f"  ✓ PASS: {msg}")


def print_fail(msg):
    print(f"  ✗ FAIL: {msg}")


def print_info(msg):
    print(f"  ℹ {msg}")


# ─────────────────────────────────────────────────────────────
# Test 1: Feature Extractor
# ─────────────────────────────────────────────────────────────
def test_feature_extractor(image_path=None):
    print_header("TEST 1: Feature Extractor")

    from models.feature_extractor import FashionFeatureExtractor

    try:
        # Test model initialization
        start = time.time()
        extractor = FashionFeatureExtractor(
            embedding_dim=512, backbone='resnet50', fine_tune_layers=0
        )
        elapsed = time.time() - start
        print_pass(f"ResNet50 extractor initialized ({elapsed:.1f}s)")

        # Test model output shape
        model = extractor.get_model()
        print_info(f"Model output shape: {model.output_shape}")
        assert model.output_shape[-1] == 512, "Wrong embedding dimension"
        print_pass("Output dimension is 512")

        # Test feature extraction
        if image_path and os.path.exists(image_path):
            features = extractor.extract_features(image_path)
            print_pass(f"Feature extraction: shape={features.shape}")
            assert features.shape == (512,), f"Expected (512,), got {features.shape}"
            assert abs(np.linalg.norm(features) - 1.0) < 0.01, "Features not normalized"
            print_pass("Features are L2 normalized")
        else:
            # Test with synthetic input
            import tensorflow as tf
            dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            output = model.predict(dummy_input, verbose=0)
            assert output.shape == (1, 512), f"Expected (1, 512), got {output.shape}"
            print_pass(f"Synthetic input: output shape={output.shape}")

        return True

    except Exception as e:
        print_fail(f"Feature extractor error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 2: Fashion Classifier
# ─────────────────────────────────────────────────────────────
def test_fashion_classifier(image_path=None):
    print_header("TEST 2: Fashion Classifier (Multi-task)")

    from models.fashion_classifier import (
        FashionClassifier, CATEGORY_LABELS, STYLE_LABELS,
        SEASON_LABELS, COLOR_LABELS
    )

    try:
        start = time.time()
        classifier = FashionClassifier()
        elapsed = time.time() - start
        print_pass(f"Classifier initialized ({elapsed:.1f}s)")

        # Check label counts
        print_info(f"Categories: {len(CATEGORY_LABELS)}")
        print_info(f"Styles: {len(STYLE_LABELS)}")
        print_info(f"Seasons: {len(SEASON_LABELS)}")
        print_info(f"Colors: {len(COLOR_LABELS)}")

        if image_path and os.path.exists(image_path):
            result = classifier.predict(image_path)
            print_pass(f"Classification result:")
            print_info(f"  Category: {result['category']} ({result['category_confidence']:.2%})")
            print_info(f"  Styles: {result['styles']}")
            print_info(f"  Seasons: {result['seasons']}")
            print_info(f"  Colors: {result['colors']}")

            assert 'category' in result, "Missing category"
            assert 'styles' in result, "Missing styles"
            assert 'seasons' in result, "Missing seasons"
            assert 'colors' in result, "Missing colors"
            print_pass("All attribute fields present in output")
        else:
            # Test with synthetic input
            model = classifier.get_model()
            dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            outputs = model.predict(dummy_input, verbose=0)
            assert len(outputs) == 4, f"Expected 4 output heads, got {len(outputs)}"
            print_pass(f"4 output heads: shapes={[o.shape for o in outputs]}")

        return True

    except Exception as e:
        print_fail(f"Classifier error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 3: Style Encoder (Triplet Loss)
# ─────────────────────────────────────────────────────────────
def test_style_encoder(image_path=None):
    print_header("TEST 3: Style Encoder (Triplet Loss)")

    from models.style_encoder import StyleEncoder

    try:
        start = time.time()
        encoder = StyleEncoder(embedding_dim=256, margin=0.5)
        elapsed = time.time() - start
        print_pass(f"Style encoder initialized ({elapsed:.1f}s)")

        # Check encoder output
        enc_model = encoder.get_encoder()
        print_info(f"Encoder output shape: {enc_model.output_shape}")

        # Check triplet training model
        train_model = encoder.get_training_model()
        print_info(f"Training model inputs: {len(train_model.inputs)}")
        assert len(train_model.inputs) == 3, "Triplet model needs 3 inputs"
        print_pass("Triplet model has 3 inputs (anchor, positive, negative)")

        if image_path and os.path.exists(image_path):
            style_embedding = encoder.encode_style(image_path)
            print_pass(f"Style embedding: shape={style_embedding.shape}")
            assert style_embedding.shape == (256,), f"Expected (256,), got {style_embedding.shape}"
            print_pass("Style embedding dimension correct (256)")

            # Test similarity
            sim = encoder.compute_style_similarity(style_embedding, style_embedding)
            assert abs(sim - 1.0) < 0.01, f"Self-similarity should be ~1.0, got {sim}"
            print_pass(f"Self-similarity: {sim:.4f} (expected ~1.0)")
        else:
            dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            output = enc_model.predict(dummy_input, verbose=0)
            assert output.shape == (1, 256), f"Expected (1, 256), got {output.shape}"
            print_pass(f"Synthetic: output shape={output.shape}")

        return True

    except Exception as e:
        print_fail(f"Style encoder error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 4: LSTM-VAE Autoencoder
# ─────────────────────────────────────────────────────────────
def test_autoencoder():
    print_header("TEST 4: LSTM-VAE Autoencoder")

    from models.autoencoder import FashionAutoencoder

    try:
        start = time.time()
        vae = FashionAutoencoder(
            input_dim=512, latent_dim=128, seq_length=32, beta=1.0
        )
        elapsed = time.time() - start
        print_pass(f"LSTM-VAE initialized ({elapsed:.1f}s)")

        # Check dimensions
        print_info(f"Input dim: 512, Latent dim: 128, Seq length: 32")
        print_info(f"Features per step: {vae.features_per_step}")

        # Test encoding
        dummy_embedding = np.random.rand(512).astype(np.float32)
        refined = vae.refine_embedding(dummy_embedding)
        assert refined.shape == (128,), f"Expected (128,), got {refined.shape}"
        print_pass(f"Single embedding refinement: {dummy_embedding.shape} -> {refined.shape}")

        # Test batch encoding
        batch = np.random.rand(10, 512).astype(np.float32)
        refined_batch = vae.refine_embeddings_batch(batch)
        assert refined_batch.shape == (10, 128), f"Expected (10, 128), got {refined_batch.shape}"
        print_pass(f"Batch refinement: {batch.shape} -> {refined_batch.shape}")

        # Test interpolation
        emb1 = np.random.rand(512).astype(np.float32)
        emb2 = np.random.rand(512).astype(np.float32)
        interpolated = vae.interpolate(emb1, emb2, steps=5)
        assert interpolated.shape == (5, 128), f"Expected (5, 128), got {interpolated.shape}"
        print_pass(f"Interpolation (5 steps): shape={interpolated.shape}")

        # Test novelty detection
        novelty_score = vae.detect_novelty(dummy_embedding)
        print_pass(f"Novelty score: {novelty_score:.4f}")

        # Test VAE training (1 epoch, small data)
        vae.compile_model(learning_rate=1e-3)
        train_data = np.random.rand(50, 512).astype(np.float32)
        history = vae.train(train_data, epochs=1, batch_size=16, validation_split=0.1)
        print_pass(f"LSTM-VAE training: loss={history.history['loss'][-1]:.4f}")

        return True

    except Exception as e:
        print_fail(f"Autoencoder error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 5: LSTM Sequence Predictor
# ─────────────────────────────────────────────────────────────
def test_sequence_predictor():
    print_header("TEST 5: LSTM Sequence Predictor")

    from recommendation.engine import SequencePredictor

    try:
        start = time.time()
        predictor = SequencePredictor(
            embedding_dim=512, max_seq_length=20, lstm_units=256
        )
        elapsed = time.time() - start
        print_pass(f"Sequence predictor initialized ({elapsed:.1f}s)")

        # Test prediction with short history
        history = [np.random.rand(512).astype(np.float32) for _ in range(5)]
        predicted = predictor.predict_next(history)
        assert predicted.shape == (512,), f"Expected (512,), got {predicted.shape}"
        print_pass(f"Prediction from 5-item history: shape={predicted.shape}")

        # Test with longer history
        long_history = [np.random.rand(512).astype(np.float32) for _ in range(25)]
        predicted_long = predictor.predict_next(long_history)
        assert predicted_long.shape == (512,), f"Expected (512,), got {predicted_long.shape}"
        print_pass("Handles history longer than max_seq_length (truncates)")

        # Test training
        predictor.compile_model(learning_rate=1e-3)
        sequences = np.random.rand(20, 20, 512).astype(np.float32)
        next_items = np.random.rand(20, 512).astype(np.float32)
        history_obj = predictor.train(sequences, next_items, epochs=1, batch_size=8)
        print_pass(f"Training: loss={history_obj.history['loss'][-1]:.4f}")

        return True

    except Exception as e:
        print_fail(f"Sequence predictor error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 6: Recommendation Engine
# ─────────────────────────────────────────────────────────────
def test_recommendation_engine():
    print_header("TEST 6: Multi-modal Recommendation Engine")

    from recommendation.engine import RecommendationEngine

    try:
        # Create synthetic catalog
        n_items = 100
        visual_features = np.random.rand(n_items, 512).astype(np.float32)
        style_features = np.random.rand(n_items, 256).astype(np.float32)
        categories = np.random.choice(
            ['Topwear', 'Bottomwear', 'Shoes', 'Bags'], size=n_items
        ).tolist()
        filenames = [f"images/item_{i}.jpg" for i in range(n_items)]

        engine = RecommendationEngine(
            visual_weight=0.35,
            style_weight=0.25,
            category_weight=0.15,
            sequence_weight=0.25,
            n_neighbors=20,
            metric='cosine'
        )

        engine.build_index(
            visual_features=visual_features,
            style_features=style_features,
            categories=categories,
            filenames=filenames
        )
        print_pass("Index built with 100 synthetic items")

        # Test multi-modal recommendation
        query_visual = np.random.rand(512).astype(np.float32)
        query_style = np.random.rand(256).astype(np.float32)

        results = engine.recommend(
            query_visual=query_visual,
            query_style=query_style,
            query_category='Topwear',
            n_results=6
        )
        assert len(results['indices']) > 0, "No results returned"
        assert len(results['scores']) == len(results['indices']), "Score/index mismatch"
        print_pass(f"Multi-modal recommendation: {len(results['indices'])} results")
        print_info(f"  Top scores: {results['scores'][:3]}")

        # Test simple recommendation
        simple_results = engine.recommend_simple(query_visual, n_results=5)
        assert len(simple_results) == 5, f"Expected 5, got {len(simple_results)}"
        print_pass(f"Simple recommendation: {len(simple_results)} results")

        # Test style-based recommendation
        style_results = engine.get_similar_by_style(query_style, n_results=5)
        assert len(style_results) == 5, f"Expected 5, got {len(style_results)}"
        print_pass(f"Style-based recommendation: {len(style_results)} results")

        # Test complementary items
        comp_results = engine.get_complementary('Topwear', query_style, n_results=5)
        print_pass(f"Complementary items: {len(comp_results)} results")
        # Verify complementary items are different category
        for idx in comp_results:
            assert categories[idx] != 'Topwear', "Complementary item should be different category"
        print_pass("All complementary items are different category")

        # Test category filter
        filtered_results = engine.recommend(
            query_visual=query_visual,
            query_style=query_style,
            query_category='Shoes',
            n_results=6,
            category_filter='Shoes'
        )
        for idx in filtered_results['indices']:
            assert categories[idx] == 'Shoes', f"Filter failed: got {categories[idx]}"
        print_pass("Category filter working correctly")

        return True

    except Exception as e:
        print_fail(f"Recommendation engine error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 7: Diversity Re-ranker
# ─────────────────────────────────────────────────────────────
def test_diversity_reranker():
    print_header("TEST 7: Diversity Re-ranker (MMR)")

    from recommendation.diversity import DiversityReranker

    try:
        reranker = DiversityReranker(lambda_param=0.6)

        # Create candidates with known similarity structure
        n_items = 50
        feature_matrix = np.random.rand(n_items, 512).astype(np.float32)
        # Make some items very similar (near-duplicates)
        feature_matrix[1] = feature_matrix[0] + 0.01 * np.random.rand(512)
        feature_matrix[2] = feature_matrix[0] + 0.02 * np.random.rand(512)

        candidate_indices = list(range(20))
        candidate_scores = [1.0 / (i + 1) for i in range(20)]

        reranked = reranker.rerank(
            candidate_indices, candidate_scores, feature_matrix, n_results=6
        )
        assert len(reranked) == 6, f"Expected 6, got {len(reranked)}"
        print_pass(f"Re-ranked 20 candidates to 6 diverse results")

        # Near-duplicates shouldn't all appear
        dup_count = sum(1 for idx in reranked if idx in [0, 1, 2])
        print_info(f"  Near-duplicate cluster: {dup_count}/3 selected")
        assert dup_count < 3, "MMR should avoid selecting all near-duplicates"
        print_pass("MMR successfully reduces near-duplicate recommendations")

        # Test diversity score
        div_score = reranker.compute_diversity_score(reranked, feature_matrix)
        print_pass(f"Diversity score: {div_score:.4f}")
        assert div_score > 0, "Diversity score should be positive"

        # Compare diversity with/without reranking
        naive_top6 = candidate_indices[:6]
        naive_div = reranker.compute_diversity_score(naive_top6, feature_matrix)
        print_info(f"  Without MMR diversity: {naive_div:.4f}")
        print_info(f"  With MMR diversity: {div_score:.4f}")

        return True

    except Exception as e:
        print_fail(f"Diversity reranker error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 8: Attribute Filters
# ─────────────────────────────────────────────────────────────
def test_attribute_filters():
    print_header("TEST 8: Attribute Filters")

    from recommendation.filters import AttributeFilter

    try:
        filter_engine = AttributeFilter()

        # Create mock metadata
        metadata = [
            {'category': 'Topwear', 'styles': ['Casual'], 'seasons': ['Summer'], 'colors': ['Blue']},
            {'category': 'Bottomwear', 'styles': ['Formal'], 'seasons': ['Winter'], 'colors': ['Black']},
            {'category': 'Shoes', 'styles': ['Sports'], 'seasons': ['All-Season'], 'colors': ['White']},
            {'category': 'Topwear', 'styles': ['Formal', 'Party'], 'seasons': ['Winter'], 'colors': ['Red']},
            {'category': 'Bags', 'styles': ['Casual'], 'seasons': ['All-Season'], 'colors': ['Brown']},
        ]
        filter_engine.set_metadata(metadata)
        print_pass("Metadata loaded for 5 items")

        candidates = [0, 1, 2, 3, 4]

        # Category filter
        cat_filtered = filter_engine.filter_by_category(candidates, 'Topwear')
        assert cat_filtered == [0, 3], f"Expected [0, 3], got {cat_filtered}"
        print_pass(f"Category filter (Topwear): {cat_filtered}")

        # Style filter
        style_filtered = filter_engine.filter_by_style(candidates, ['Casual'])
        assert 0 in style_filtered and 4 in style_filtered
        print_pass(f"Style filter (Casual): {style_filtered}")

        # Season filter
        season_filtered = filter_engine.filter_by_season(candidates, ['Winter'])
        assert 1 in season_filtered and 3 in season_filtered
        # All-Season items should also match
        assert 2 in season_filtered and 4 in season_filtered
        print_pass(f"Season filter (Winter + All-Season): {season_filtered}")

        # Color filter
        color_filtered = filter_engine.filter_by_color(candidates, ['Blue', 'Red'])
        assert color_filtered == [0, 3], f"Expected [0, 3], got {color_filtered}"
        print_pass(f"Color filter (Blue, Red): {color_filtered}")

        # Exclude category
        excluded = filter_engine.filter_exclude_category(candidates, 'Topwear')
        assert 0 not in excluded and 3 not in excluded
        print_pass(f"Exclude category (Topwear): {excluded}")

        # Combined filters
        combined = filter_engine.apply_filters(candidates, {
            'category': 'Topwear',
            'styles': ['Formal']
        })
        assert combined == [3], f"Expected [3], got {combined}"
        print_pass(f"Combined filter (Topwear + Formal): {combined}")

        return True

    except Exception as e:
        print_fail(f"Attribute filter error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  PERSONALIZED FASHION ADVISER - DEEP LEARNING TEST SUITE  ║")
    print("╚" + "═" * 58 + "╝")

    if args.image:
        if os.path.exists(args.image):
            print(f"\nTest image: {args.image}")
        else:
            print(f"\nWarning: Image '{args.image}' not found. Using synthetic data.")
            args.image = None
    else:
        print("\nNo test image specified (use --image). Using synthetic data.")

    results = {}
    total_start = time.time()

    if args.test_mode == 'full':
        # Model tests (slower, require TF)
        results['Feature Extractor'] = test_feature_extractor(args.image)
        results['Fashion Classifier'] = test_fashion_classifier(args.image)
        results['Style Encoder'] = test_style_encoder(args.image)
        results['LSTM-VAE Autoencoder'] = test_autoencoder()
        results['LSTM Sequence Predictor'] = test_sequence_predictor()
    else:
        print("\n  [Quick mode: skipping model instantiation tests]")

    # These are fast and always run
    results['Recommendation Engine'] = test_recommendation_engine()
    results['Diversity Re-ranker'] = test_diversity_reranker()
    results['Attribute Filters'] = test_attribute_filters()

    # Summary
    total_elapsed = time.time() - total_start
    print_header("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  Results: {passed}/{total} passed")
    print(f"  Time: {total_elapsed:.1f}s")

    if passed == total:
        print("\n  🎉 All tests passed! Pipeline is ready.")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Check output above.")

    return 0 if passed == total else 1


if __name__ == '__main__':
    exit(main())
