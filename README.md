# 👗 Personalized Fashion Adviser

A deep learning-powered visual search and recommendation engine for fashion. Upload an image and get intelligent, style-aware recommendations using multi-modal AI.

## What's New — Deep Learning Enhancements

This project has been upgraded from a basic ResNet50 + KNN system to a full deep learning pipeline:

| Component | Before | After |
|-----------|--------|-------|
| Feature Extraction | ResNet50 (frozen) | Dual-backbone (ResNet50 + EfficientNetB3) with learnable projection head |
| Similarity | Euclidean KNN | Multi-modal weighted scoring (visual + style + LSTM sequence) |
| Classification | None | Multi-task: category, style, season, color |
| Style Matching | None | Triplet-loss trained style encoder |
| Embedding Quality | Raw CNN features | LSTM-VAE refined embeddings |
| User Modeling | None | LSTM sequence predictor learns browsing patterns |
| Result Quality | Top-K nearest | MMR diversity re-ranking + attribute filtering |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                                │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  ResNet50 +  │   Triplet    │  Multi-task  │   LSTM-VAE     │
│ EfficientNet │    Loss      │  Classifier  │  Autoencoder   │
│  (Visual)    │  (Style)     │ (Attributes) │  (Refinement)  │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              Multi-Modal Recommendation Engine                │
│    ┌────────────────────────────────────────────────┐        │
│    │  LSTM Sequence Predictor (user history)        │        │
│    │  Weighted Scoring (visual + style + sequence)  │        │
│    │  MMR Diversity Re-ranking                      │        │
│    │  Attribute Filtering (category/style/season)   │        │
│    └────────────────────────────────────────────────┘        │
├──────────────────────────────────────────────────────────────┤
│              RANKED, DIVERSE RECOMMENDATIONS                  │
└──────────────────────────────────────────────────────────────┘
```

## Deep Learning Models

### 1. Feature Extractor (`models/feature_extractor.py`)
- **Dual backbone**: ResNet50 (structural features) + EfficientNetB3 (texture/patterns)
- **Projection head**: Dense layers with L2 normalization for compact 512-d embeddings
- **Fine-tuning support**: Top N layers unfrozen for domain adaptation

### 2. Fashion Classifier (`models/fashion_classifier.py`)
- **Multi-task heads**: Predicts category, style, season, and color simultaneously
- **20 categories**: Topwear, Bottomwear, Shoes, Bags, Dress, Saree, etc.
- **8 styles**: Casual, Formal, Sports, Ethnic, Party, Streetwear, etc.
- **Shared backbone** with task-specific dense heads

### 3. Style Encoder (`models/style_encoder.py`)
- **Triplet loss training**: Learns style similarity beyond pixel appearance
- **256-d style embeddings**: Captures "vibe" — two items can look different but match in style
- **Use case**: A blue formal shirt and black formal pants have high style similarity

### 4. LSTM-VAE Autoencoder (`models/autoencoder.py`)
- **Bidirectional LSTM encoder**: Captures sequential dependencies between feature groups
- **Variational latent space**: Smooth, continuous representation for interpolation
- **Embedding refinement**: Denoises and compresses to 128-d latent codes
- **Novelty detection**: High reconstruction error flags unusual items

### 5. LSTM Sequence Predictor (`recommendation/engine.py`)
- **User behavior modeling**: Learns patterns from browsing history
- **Predicts next preferred item**: "Users who looked at X, Y, Z tend to want W next"
- **Bidirectional LSTM**: Captures temporal patterns in shopping sessions

## Tech Stack

- **TensorFlow / Keras** — Deep learning models (CNN, LSTM, VAE, Triplet Loss)
- **Streamlit** — Interactive web UI
- **scikit-learn** — Nearest neighbor search, metrics
- **NumPy** — Numerical computation
- **Pillow / OpenCV** — Image processing

## Project Structure

```
├── main.py                     # Streamlit web application
├── app.py                      # Embedding generation pipeline (CLI)
├── test.py                     # Comprehensive test suite
├── requirements.txt            # Python dependencies
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py    # Dual-backbone CNN feature extraction
│   ├── fashion_classifier.py   # Multi-task category/style/season/color
│   ├── style_encoder.py        # Triplet-loss style embeddings
│   └── autoencoder.py          # LSTM-VAE embedding refinement
├── recommendation/
│   ├── __init__.py
│   ├── engine.py               # Multi-modal engine + LSTM sequence predictor
│   ├── diversity.py            # MMR diversity re-ranking
│   └── filters.py              # Attribute-based filtering
├── images/                     # Product catalog images
├── uploads/                    # User uploaded images
├── saved_models/               # Trained model weights
├── embeddings.pkl              # Visual embeddings
├── style_embeddings.pkl        # Style embeddings
├── refined_embeddings.pkl      # VAE-refined embeddings
├── metadata.pkl                # Classification metadata
├── categories.pkl              # Category labels
└── filenames.pkl               # Image file paths
```

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the fashion dataset and place images in the `images/` directory:
- Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

### 3. Generate Embeddings

```bash
# Basic (ResNet50 only, fast)
python app.py --images_dir images

# Full pipeline with LSTM-VAE training
python app.py --images_dir images --backbone resnet50 --train_vae

# Dual backbone (highest quality, slower)
python app.py --images_dir images --backbone dual --train_vae --batch_size 16
```

**CLI Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--images_dir` | `images` | Directory with product images |
| `--backbone` | `resnet50` | `resnet50`, `efficientnet`, or `dual` |
| `--embedding_dim` | `512` | Visual embedding dimension |
| `--style_dim` | `256` | Style embedding dimension |
| `--latent_dim` | `128` | VAE latent dimension |
| `--batch_size` | `32` | Processing batch size |
| `--train_vae` | off | Train the LSTM-VAE on embeddings |
| `--vae_epochs` | `50` | VAE training epochs |
| `--skip_style` | off | Skip style embedding extraction |
| `--skip_classifier` | off | Skip classification |

### 4. Run the App

```bash
streamlit run main.py
```

### 5. Run Tests

```bash
# Full test suite (requires TensorFlow)
python test.py

# With a specific test image
python test.py --image sample/shirt.jpg

# Quick mode (skips model instantiation)
python test.py --test_mode quick
```

## App Features

- **4 recommendation modes**: Smart (multi-modal), Visual Only, Style Match, Complementary Items
- **AI analysis panel**: Shows predicted category, style, season, and colors
- **Diversity slider**: Control variety vs. similarity in results
- **Attribute filters**: Filter by category, style, season in the sidebar
- **Session tracking**: LSTM learns from your browsing within a session
- **Complete the Look**: Suggests items from complementary categories

## How the LSTM Components Work

### LSTM-VAE (Embedding Refinement)
The autoencoder reshapes a 512-d embedding into a sequence of 32 steps (16 features each), then processes it with Bidirectional LSTM layers. This captures relationships between feature groups that dense layers miss — like how color features relate to texture features in fashion items.

### LSTM Sequence Predictor (User Modeling)
Takes the last 20 items a user interacted with (as embeddings), processes the sequence with Bidirectional LSTM, and predicts what embedding they'd prefer next. This captures patterns like "browsed winter coats → scarves → boots" to proactively suggest the next logical item.

## Screenshots

![Screenshot (1326)](https://github.com/Kushmathur1206/Fashion-Recommendation-System/assets/99969817/b7216fc2-b23e-4c4a-9aa0-0d4d7938200e)
![Screenshot (1327)](https://github.com/Kushmathur1206/Fashion-Recommendation-System/assets/99969817/322458e1-12db-4730-87fe-64b099885168)

## License

MIT
