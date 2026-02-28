"""
Linear Probing: The standard way to evaluate self-supervised models.

After pre-training, we:
1. Freeze the encoder (no more changes to it)
2. Add a simple linear classification layer on top
3. Train ONLY the classification layer on a small labeled dataset
4. Measure accuracy

Good results here = the encoder learned useful representations.

Supports both:
- Single-label classification (CrossEntropyLoss) — most datasets
- Multi-label classification (BCEWithLogitsLoss) — e.g. ChestXray14
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    average_precision_score,
)
from medjepa.utils.device import get_device


class LinearProbe(nn.Module):
    """A single linear layer for classification."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LinearProbeEvaluator:
    """
    Evaluates a pre-trained model using linear probing.

    Supports both single-label (softmax + CE) and multi-label (sigmoid + BCE)
    classification.  Set ``multi_label=True`` for datasets like ChestXray14
    where each image can have multiple disease labels simultaneously.
    """

    def __init__(
        self,
        pretrained_model,
        num_classes: int,
        embed_dim: int = 768,
        multi_label: bool = False,
    ):
        self.device = get_device()
        self.pretrained_model = pretrained_model.to(self.device)
        self.pretrained_model.eval()
        self.multi_label = multi_label

        # Freeze the model — we don't want to change it
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Create the linear probe
        self.probe = LinearProbe(embed_dim, num_classes).to(self.device)

    def extract_features(self, dataloader: DataLoader) -> tuple:
        """
        Run all images through the frozen encoder to get embeddings.
        This only needs to be done ONCE.
        """
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)

                # Get embeddings from the frozen encoder
                features = self.pretrained_model.encode(images)

                all_features.append(features.cpu())
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return features, labels

    def train_probe(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
    ):
        """Train the linear classification layer."""
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(train_features, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(self.probe.parameters(), lr=lr, momentum=0.9)

        if self.multi_label:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        self.probe.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.probe(features)

                if self.multi_label:
                    # labels should be float for BCE
                    loss = criterion(logits, labels.float())
                else:
                    loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Probe Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    def evaluate(
        self,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> dict:
        """Evaluate the linear probe on test data."""
        self.probe.eval()
        with torch.no_grad():
            test_features = test_features.to(self.device)
            logits = self.probe(test_features)

        if self.multi_label:
            return self._evaluate_multi_label(logits, test_labels)
        else:
            return self._evaluate_single_label(logits, test_labels)

    def _evaluate_single_label(self, logits: torch.Tensor, test_labels: torch.Tensor) -> dict:
        """Standard single-label evaluation with accuracy + AUC."""
        predictions = logits.argmax(dim=1).cpu().numpy()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        true_labels = test_labels.numpy()

        results = {
            "accuracy": float(accuracy_score(true_labels, predictions)),
            "report": classification_report(
                true_labels, predictions, output_dict=True,
            ),
        }

        # AUC if binary or we can compute it
        try:
            if probabilities.shape[1] == 2:
                results["auc"] = float(roc_auc_score(true_labels, probabilities[:, 1]))
            else:
                results["auc"] = float(roc_auc_score(
                    true_labels, probabilities, multi_class="ovr", average="macro",
                ))
        except Exception:
            results["auc"] = None

        return results

    def _evaluate_multi_label(self, logits: torch.Tensor, test_labels: torch.Tensor) -> dict:
        """
        Multi-label evaluation with per-class AUC and mAP.

        For multi-label problems (e.g. ChestXray14), accuracy is replaced
        by mean AUC and mean Average Precision across all disease labels.
        """
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        true_labels = test_labels.numpy().astype(int)

        n_classes = probs.shape[1]

        # Per-class AUC
        per_class_auc = {}
        for c in range(n_classes):
            try:
                per_class_auc[c] = float(roc_auc_score(true_labels[:, c], probs[:, c]))
            except ValueError:
                per_class_auc[c] = None  # only one class present

        valid_aucs = [v for v in per_class_auc.values() if v is not None]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None

        # Mean Average Precision
        try:
            mean_ap = float(average_precision_score(true_labels, probs, average="macro"))
        except Exception:
            mean_ap = None

        # Exact-match accuracy (all labels correct for a sample)
        exact_match = float((preds == true_labels).all(axis=1).mean())

        # Per-sample accuracy (fraction of correct labels per sample)
        sample_acc = float((preds == true_labels).mean())

        results = {
            "accuracy": sample_acc,
            "exact_match_accuracy": exact_match,
            "auc": mean_auc,
            "per_class_auc": per_class_auc,
            "mean_average_precision": mean_ap,
        }

        return results
