"""
LSTM-based Variational Autoencoder for Embedding Refinement.

Uses LSTM layers to capture sequential patterns in fashion embeddings,
enabling the model to learn temporal/contextual relationships between
fashion features (e.g., patterns in how colors flow into shapes).

Benefits:
1. LSTM captures sequential dependencies in feature dimensions
2. Denoising - removes irrelevant noise from embeddings
3. Smooth latent space enables meaningful style interpolation
4. Sequence-aware encoding preserves structural relationships
"""

import numpy as np
import tensorflow as tf
import keras.ops as ops
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Lambda,
    LSTM, RepeatVector, TimeDistributed, Reshape, Bidirectional
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Sampling(tf.keras.layers.Layer):
    """Reparameterization trick: sample from N(z_mean, z_var)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class FashionAutoencoder:
    """
    LSTM-based Variational Autoencoder for fashion embedding refinement.

    Architecture:
    - Encoder: Reshapes embedding into a sequence, processes with
      Bidirectional LSTM to capture cross-feature dependencies,
      then maps to a smooth latent space
    - Decoder: Uses LSTM to reconstruct the embedding sequence
      from the latent code

    The LSTM layers learn relationships between groups of features
    (e.g., color features relate to texture features in specific ways),
    producing richer, more structured refined embeddings.
    """

    def __init__(self, input_dim=512, latent_dim=128, seq_length=32, beta=1.0):
        """
        Args:
            input_dim: Dimension of input embeddings (must be divisible by seq_length)
            latent_dim: Dimension of the latent (refined) embedding
            seq_length: Number of time steps for LSTM (input_dim / seq_length = features per step)
            beta: Weight of KL divergence loss (beta-VAE)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.features_per_step = input_dim // seq_length
        self.beta = beta
        self.encoder, self.decoder, self.vae = self._build_model()

    def _build_model(self):
        """Build the LSTM-VAE: encoder + decoder."""
        # === LSTM Encoder ===
        encoder_input = Input(shape=(self.input_dim,), name='encoder_input')

        # Reshape flat embedding into sequence for LSTM processing
        # Each "time step" represents a group of related features
        x = Reshape(
            (self.seq_length, self.features_per_step), name='reshape_to_seq'
        )(encoder_input)

        # Bidirectional LSTM captures forward and backward feature dependencies
        x = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            name='encoder_bilstm1'
        )(x)
        x = Bidirectional(
            LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
            name='encoder_bilstm2'
        )(x)

        # Dense compression before latent space
        x = Dense(256, activation='relu', name='enc_dense')(x)
        x = BatchNormalization(name='enc_bn')(x)
        x = Dropout(0.2, name='enc_dropout')(x)

        # Latent space parameters
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # Sample from latent space
        z = Sampling(name='sampling')([z_mean, z_log_var])

        encoder = Model(
            encoder_input, [z_mean, z_log_var, z], name='lstm_vae_encoder'
        )

        # === LSTM Decoder ===
        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')

        # Expand latent code back to sequence
        y = Dense(256, activation='relu', name='dec_dense1')(decoder_input)
        y = BatchNormalization(name='dec_bn1')(y)
        y = RepeatVector(self.seq_length, name='repeat_to_seq')(y)

        # LSTM decoder reconstructs the feature sequence
        y = LSTM(128, return_sequences=True, dropout=0.2, name='decoder_lstm1')(y)
        y = LSTM(64, return_sequences=True, dropout=0.2, name='decoder_lstm2')(y)

        # Map each time step back to feature dimension
        y = TimeDistributed(
            Dense(self.features_per_step, activation='linear'),
            name='dec_time_dense'
        )(y)

        # Flatten back to original embedding shape
        decoder_output = Reshape(
            (self.input_dim,), name='reshape_to_flat'
        )(y)

        decoder = Model(decoder_input, decoder_output, name='lstm_vae_decoder')

        # === Full VAE ===
        z_mean_out, z_log_var_out, z_out = encoder(encoder_input)
        reconstructed = decoder(z_out)

        vae = Model(encoder_input, reconstructed, name='fashion_lstm_vae')

        # KL divergence regularization via custom layer
        class KLDivergenceLayer(tf.keras.layers.Layer):
            def __init__(self, beta, **kwargs):
                super().__init__(**kwargs)
                self.beta = beta

            def call(self, inputs):
                z_mean, z_log_var = inputs
                kl_loss = -0.5 * ops.mean(
                    1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)
                )
                self.add_loss(self.beta * kl_loss)
                return z_mean

        kl_layer = KLDivergenceLayer(beta=self.beta, name='kl_divergence')
        kl_layer([z_mean_out, z_log_var_out])

        return encoder, decoder, vae

    def compile_model(self, learning_rate=1e-3):
        """Compile the LSTM-VAE with reconstruction + KL loss."""
        self.vae.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )

    def refine_embedding(self, embedding):
        """
        Refine a single embedding through the LSTM encoder.
        Returns the latent mean (deterministic at inference).

        Args:
            embedding: Input embedding vector of shape (input_dim,)

        Returns:
            Refined embedding of shape (latent_dim,)
        """
        embedding = np.expand_dims(embedding, axis=0)
        z_mean, _, _ = self.encoder.predict(embedding, verbose=0)
        return z_mean.flatten()

    def refine_embeddings_batch(self, embeddings):
        """
        Refine a batch of embeddings through LSTM encoder.

        Args:
            embeddings: Array of shape (n, input_dim)

        Returns:
            Refined embeddings of shape (n, latent_dim)
        """
        z_mean, _, _ = self.encoder.predict(embeddings, verbose=0)
        return z_mean

    def interpolate(self, embedding1, embedding2, steps=10):
        """
        Interpolate between two items in latent space.
        Useful for style exploration ("show me items between A and B").

        Args:
            embedding1, embedding2: Input embeddings
            steps: Number of interpolation steps

        Returns:
            Array of interpolated latent embeddings
        """
        emb1 = np.expand_dims(embedding1, axis=0)
        emb2 = np.expand_dims(embedding2, axis=0)

        z_mean1, _, _ = self.encoder.predict(emb1, verbose=0)
        z_mean2, _, _ = self.encoder.predict(emb2, verbose=0)

        # Linear interpolation in latent space (smooth due to VAE)
        alphas = np.linspace(0, 1, steps)
        interpolated = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z_mean1 + alpha * z_mean2
            interpolated.append(z_interp.flatten())

        return np.array(interpolated)

    def detect_novelty(self, embedding, threshold=None):
        """
        Detect novel/unusual items via reconstruction error.
        LSTM's sequential processing makes this sensitive to
        unusual feature combinations.

        Args:
            embedding: Input embedding
            threshold: Novelty threshold (if None, returns raw score)

        Returns:
            Novelty score and boolean if threshold given
        """
        emb = np.expand_dims(embedding, axis=0)
        reconstructed = self.vae.predict(emb, verbose=0)
        reconstruction_error = np.mean(np.square(emb - reconstructed))

        if threshold is not None:
            return reconstruction_error, reconstruction_error > threshold
        return reconstruction_error

    def train(self, embeddings, epochs=50, batch_size=64, validation_split=0.1):
        """
        Train the LSTM-VAE on a collection of embeddings.

        Args:
            embeddings: Array of shape (n, input_dim)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation

        Returns:
            Training history
        """
        history = self.vae.fit(
            embeddings, embeddings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
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

    def save_model(self, path='saved_models/autoencoder'):
        """Save encoder and decoder weights."""
        self.encoder.save_weights(f'{path}_lstm_encoder.weights.h5')
        self.decoder.save_weights(f'{path}_lstm_decoder.weights.h5')

    def load_model(self, path='saved_models/autoencoder'):
        """Load pretrained weights."""
        self.encoder.load_weights(f'{path}_lstm_encoder.weights.h5')
        self.decoder.load_weights(f'{path}_lstm_decoder.weights.h5')

    def get_encoder(self):
        """Return encoder model."""
        return self.encoder

    def get_decoder(self):
        """Return decoder model."""
        return self.decoder
