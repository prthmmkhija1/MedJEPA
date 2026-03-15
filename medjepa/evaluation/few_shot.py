"""
Few-Shot Learning Evaluation.

Tests: Can the model learn from very few examples?

Example: Given only 5 examples of "pneumonia" and 5 examples of "normal",
can the model correctly classify new chest X-rays?

This is the most clinically relevant evaluation because in real hospitals,
getting 5 labeled examples is feasible, but 50,000 is not.
"""

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sklearn_normalize
from typing import Tuple
from medjepa.utils.device import get_device


def _l2_normalize(features):
    """L2-normalize feature vectors (works on numpy arrays)."""
    return sklearn_normalize(features, norm='l2', axis=1)


class FewShotEvaluator:
    """
    Evaluate representations using few-shot classification.

    Method: k-Nearest Neighbors (kNN) with cosine similarity
    - Encode all images to embeddings
    - L2-normalize embeddings so cosine similarity = dot product
    - For each test image, find the k closest training embeddings
    - Classify based on majority vote of those k neighbors
    """

    def __init__(self, pretrained_model, k: int = 5):
        self.device = get_device()
        self.model = pretrained_model.to(self.device)
        self.model.eval()
        self.k = k

    def evaluate_n_shot(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        n_shot: int = 5,
    ) -> dict:
        """
        Evaluate n-shot classification using cosine-similarity kNN.
        """
        with torch.no_grad():
            support_features = self.model.encode(
                support_images.to(self.device)
            ).cpu().numpy()
            query_features = self.model.encode(
                query_images.to(self.device)
            ).cpu().numpy()

        # L2-normalize for cosine similarity
        support_features = _l2_normalize(support_features)
        query_features = _l2_normalize(query_features)

        knn = KNeighborsClassifier(
            n_neighbors=max(1, min(self.k, len(support_labels))),
            metric='cosine',
            algorithm='brute',
        )
        knn.fit(support_features, support_labels.numpy())

        predictions = knn.predict(query_features)
        accuracy = accuracy_score(query_labels.numpy(), predictions)

        return {
            "n_shot": n_shot,
            "accuracy": accuracy,
            "num_support": len(support_labels),
            "num_query": len(query_labels),
        }

    def evaluate_data_efficiency(
        self,
        full_train_features: torch.Tensor,
        full_train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        fractions: list = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        max_test_samples: int = 20000,
    ) -> list:
        """
        Test with different amounts of labeled data using cosine-similarity kNN.
        Answers: "How much does performance improve as we add more labels?"
        """
        results = []
        rng = np.random.RandomState(42)

        # Subsample test set if too large
        if len(test_labels) > max_test_samples:
            test_idx = rng.choice(len(test_labels), max_test_samples, replace=False)
            test_features = test_features[test_idx]
            test_labels = test_labels[test_idx]

        # Convert and L2-normalize once
        test_np = _l2_normalize(test_features.numpy())

        for frac in fractions:
            n = max(1, int(len(full_train_labels) * frac))
            max_train = min(n, 50000)
            indices = rng.choice(len(full_train_labels), max_train, replace=False)

            subset_features = _l2_normalize(full_train_features[indices].numpy())
            subset_labels = full_train_labels[indices].numpy()

            n_neighbors = max(1, min(self.k, len(subset_labels)))

            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric='cosine',
                algorithm='brute',
            )
            knn.fit(subset_features, subset_labels)
            predictions = knn.predict(test_np)
            accuracy = accuracy_score(test_labels.numpy(), predictions)

            results.append({
                "fraction": frac,
                "num_labeled": n,
                "accuracy": accuracy,
            })
            print(f"  {frac*100:.0f}% data ({n} samples): Accuracy = {accuracy:.4f}")

        return results


class PrototypeNetworkEvaluator:
    """
    Few-shot evaluation using prototype networks.

    For each class, compute the mean embedding (prototype) from the support set.
    Classify query samples by nearest prototype using cosine similarity.
    More robust than kNN for very-low-shot settings (e.g. 1-shot, 5-shot).
    """

    def __init__(self, pretrained_model):
        self.device = get_device()
        self.model = pretrained_model.to(self.device)
        self.model.eval()

    def evaluate_n_shot(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        query_features: np.ndarray,
        query_labels: np.ndarray,
        n_shot: int = 5,
    ) -> dict:
        """
        Evaluate n-shot classification using class prototypes.

        Args:
            support_features: Pre-extracted support embeddings (N_support, D)
            support_labels: Support labels (N_support,)
            query_features: Pre-extracted query embeddings (N_query, D)
            query_labels: Query labels (N_query,)
        """
        # L2-normalize
        support_features = _l2_normalize(support_features)
        query_features = _l2_normalize(query_features)

        # Compute class prototypes (mean of normalized support embeddings per class)
        classes = np.unique(support_labels)
        prototypes = np.zeros((len(classes), support_features.shape[1]), dtype=np.float32)
        class_to_idx = {}
        for i, cls in enumerate(classes):
            mask = support_labels == cls
            prototypes[i] = support_features[mask].mean(axis=0)
            class_to_idx[i] = cls
        # Re-normalize prototypes
        prototypes = _l2_normalize(prototypes)

        # Classify by cosine similarity (= dot product after L2 norm)
        similarities = query_features @ prototypes.T  # (N_query, N_classes)
        pred_indices = similarities.argmax(axis=1)
        predictions = np.array([class_to_idx[i] for i in pred_indices])

        accuracy = accuracy_score(query_labels, predictions)

        return {
            "n_shot": n_shot,
            "accuracy": accuracy,
            "num_support": len(support_labels),
            "num_query": len(query_labels),
            "method": "prototype_network",
        }
