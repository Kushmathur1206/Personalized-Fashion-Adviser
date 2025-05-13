# Personalized-Fashion-Adviser
A simple deep learning-based image search engine tailored for emerging fashion brands. This system enables brands to integrate visual similarity search on their own websites by generating and comparing image embeddings from their product catalog.
Key Features:
- Lets users upload an image and find visually similar clothing items.
- Helps brands enhance product discovery and engagement.
- Easily adaptable to new catalogs by embedding any custom collection.

Tech Stack
Frontend: 
- Streamlit for a lightweight and interactive web interface
Backend:
- TensorFlow + Keras for deep learning model
- ResNet50 (pretrained on ImageNet) for feature extraction
- GlobalMaxPooling2D to convert deep features into compact embeddings

Similarity Matching:
- scikit-learn's NearestNeighbors with Euclidean distance

DATASET LINK - https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
![Screenshot (1326)](https://github.com/Kushmathur1206/Fashion-Recommendation-System/assets/99969817/b7216fc2-b23e-4c4a-9aa0-0d4d7938200e)
![Screenshot (1327)](https://github.com/Kushmathur1206/Fashion-Recommendation-System/assets/99969817/322458e1-12db-4730-87fe-64b099885168)
