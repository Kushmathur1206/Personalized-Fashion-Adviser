"""
Diversity-Aware Re-ranking Module.

Ensures recommendations aren't all near-duplicates by using
Maximal Marginal Relevance (MMR) to balance relevance with diversity.

Without this, you'd get 6 nearly identical items. With it, you get
varied but still relevant suggestions.
"""

import numpy as np
from numpy.linalg import norm


class DiversityReranker:
    """
    Implements Maximal Marginal Relevance (MMR) for diversifying
    recommendation results.

    MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected

    Higher lambda = more relevance-focused
    Lower lambda = more diversity-focused
    """

    def __init__(self, lambda_param=0.6):
        """
        Args:
            lambda_param: Trade-off between relevance and diversity (0 to 1)
                          0.6 = slightly favoring relevance while maintaining diversity
        """
        self.lambda_param = lambda_param

    def rerank(self, candidate_indices, candidate_scores, feature_matrix, n_results=6):
        """
        Re-rank candidates using MMR to ensure diversity.

        Args:
            candidate_indices: List of candidate item indices
            candidate_scores: List of relevance scores (higher = more relevant)
            feature_matrix: Full feature matrix (n_items, dim) for computing diversity
            n_results: Number of diverse results to return

        Returns:
            List of re-ranked indices balancing relevance and diversity
        """
        if len(candidate_indices) <= n_results:
            return candidate_indices

        # Normalize scores to [0, 1]
        scores = np.array(candidate_scores)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores)

        selected = []
        remaining = list(range(len(candidate_indices)))

        # Greedy MMR selection
        for _ in range(n_results):
            if not remaining:
                break

            best_mmr_score = -np.inf
            best_idx = remaining[0]

            for i in remaining:
                relevance = scores[i]

                # Compute max similarity to already selected items
                if selected:
                    item_feature = feature_matrix[candidate_indices[i]]
                    max_sim = 0.0
                    for s in selected:
                        selected_feature = feature_matrix[candidate_indices[s]]
                        sim = self._cosine_similarity(item_feature, selected_feature)
                        max_sim = max(max_sim, sim)
                else:
                    max_sim = 0.0

                # MMR score
                mmr_score = (self.lambda_param * relevance -
                             (1 - self.lambda_param) * max_sim)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidate_indices[i] for i in selected]

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norms = norm(a) * norm(b)
        if norms == 0:
            return 0.0
        return dot / norms

    def compute_diversity_score(self, indices, feature_matrix):
        """
        Compute the average pairwise diversity of a set of recommendations.
        Useful for evaluating recommendation quality.

        Args:
            indices: List of item indices
            feature_matrix: Feature matrix

        Returns:
            Average diversity score (higher = more diverse)
        """
        if len(indices) < 2:
            return 0.0

        total_distance = 0.0
        pairs = 0

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                feat_i = feature_matrix[indices[i]]
                feat_j = feature_matrix[indices[j]]
                distance = 1.0 - self._cosine_similarity(feat_i, feat_j)
                total_distance += distance
                pairs += 1

        return total_distance / pairs if pairs > 0 else 0.0
