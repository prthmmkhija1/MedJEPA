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
from typing import Tuple
from medjepa.utils.device import get_device


class FewShotEvaluator:
    """
    Evaluate representations using few-shot classification.

    Method: k-Nearest Neighbors (kNN)
    - Encode all images to embeddings
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
        support_images: torch.Tensor,  # The few labeled examples
        support_labels: torch.Tensor,
        query_images: torch.Tensor,     # Test images
        query_labels: torch.Tensor,
        n_shot: int = 5,                # How many examples per class
    ) -> dict:
        """
        Evaluate n-shot classification.

        Args:
            support_images: Few labeled examples, shape (n_classes * n_shot, C, H, W)
            support_labels: Their labels
            query_images: Test images to classify
            query_labels: True labels for test images
            n_shot: Number of examples per class
        """
        with torch.no_grad():
            # Encode support and query images
            support_features = self.model.encode(
                support_images.to(self.device)
            ).cpu().numpy()
            query_features = self.model.encode(
                query_images.to(self.device)
            ).cpu().numpy()

        # Use kNN classifier
        knn = KNeighborsClassifier(n_neighbors=min(self.k, len(support_labels)))
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
    ) -> list:
        """
        Test with different amounts of labeled data.
        Answers: "How much does performance improve as we add more labels?"

        Args:
            fractions: What percentage of labels to use (0.01 = 1%, 0.5 = 50%)
        """
        results = []

        for frac in fractions:
            n = max(1, int(len(full_train_labels) * frac))
            indices = np.random.choice(len(full_train_labels), n, replace=False)

            subset_features = full_train_features[indices].numpy()
            subset_labels = full_train_labels[indices].numpy()

            knn = KNeighborsClassifier(
                n_neighbors=min(self.k, len(subset_labels))
            )
            knn.fit(subset_features, subset_labels)
            predictions = knn.predict(test_features.numpy())
            accuracy = accuracy_score(test_labels.numpy(), predictions)

            results.append({
                "fraction": frac,
                "num_labeled": n,
                "accuracy": accuracy,
            })
            print(f"  {frac*100:.0f}% data ({n} samples): Accuracy = {accuracy:.4f}")

        return results
