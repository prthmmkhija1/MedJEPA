"""
Full Fine-Tuning Evaluation & ImageNet Pretrained Baseline.

Two additional evaluation strategies beyond linear probing:

1. **FineTuneEvaluator** — unfreeze the entire encoder and train end-to-end
   with a low learning rate.  This shows the *upper bound* of what the
   pre-trained representations can achieve with full adaptation.

2. **ImageNetBaselineEvaluator** — load a torchvision ViT (or ResNet)
   pre-trained on ImageNet-1k and run the same linear-probe / fine-tune
   evaluation.  This provides a strong *baseline* to show that MedJEPA's
   self-supervised medical pre-training adds value over generic ImageNet
   features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from medjepa.utils.device import get_device


# ===================================================================
# Full Fine-Tuning Evaluator
# ===================================================================

class FineTuneEvaluator:
    """
    Evaluate a pre-trained model by fine-tuning the *entire* encoder
    together with a classification head.

    Unlike linear probing (which freezes the encoder), this:
    - Uses a small learning rate for the encoder (``encoder_lr``).
    - Uses a larger learning rate for the new head (``head_lr``).
    - Trains for ``num_epochs`` on the labeled data.
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        num_classes: int,
        embed_dim: int = 768,
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-3,
        num_epochs: int = 30,
        batch_size: int = 64,
        weight_decay: float = 0.01,
    ):
        self.device = get_device()
        self.model = pretrained_model.to(self.device)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        ).to(self.device)

    def extract_features(self, dataloader: DataLoader):
        """
        Extract features (used if you want to compare feature quality
        before/after fine-tuning).
        """
        all_features, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.model.encode(images)
                all_features.append(features.cpu())
                all_labels.append(labels)
        return torch.cat(all_features), torch.cat(all_labels)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """
        Fine-tune encoder + head end-to-end.

        Returns:
            dict with training history (losses, val accuracies).
        """
        # Unfreeze encoder
        for p in self.model.parameters():
            p.requires_grad = True

        # Separate param groups with different LRs
        encoder = self.model.encoder if hasattr(self.model, "encoder") else self.model
        encoder_params = list(encoder.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.encoder_lr},
            {"params": head_params, "lr": self.head_lr},
        ], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs,
        )
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_acc": []}

        for epoch in range(self.num_epochs):
            # --- Train ---
            self.model.train()
            self.head.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()
                features = self.model.encode(images)  # (B, D)
                logits = self.head(features)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder_params) + head_params, max_norm=1.0,
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            # --- Validate ---
            if val_loader is not None:
                val_acc = self._evaluate_loader(val_loader)
                history["val_acc"].append(val_acc)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  FT Epoch {epoch+1}/{self.num_epochs} | "
                          f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  FT Epoch {epoch+1}/{self.num_epochs} | "
                          f"Loss: {avg_loss:.4f}")

        # Re-freeze encoder after fine-tuning to be safe
        for p in self.model.parameters():
            p.requires_grad = False

        return history

    def _evaluate_loader(self, loader: DataLoader) -> float:
        self.model.eval()
        self.head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                features = self.model.encode(images)
                logits = self.head(features)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        return float(accuracy_score(labels, preds))

    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> dict:
        """
        Run evaluation on a held-out test set after fine-tuning.

        Returns dict with accuracy, AUC, classification report.
        """
        self.model.eval()
        self.head.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                features = self.model.encode(images)
                logits = self.head(features)
                probs = torch.softmax(logits, dim=1)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels)

        preds = torch.cat(all_preds).numpy()
        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()

        results = {
            "accuracy": float(accuracy_score(labels, preds)),
            "report": classification_report(labels, preds, output_dict=True),
        }

        try:
            if probs.shape[1] == 2:
                results["auc"] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                results["auc"] = float(roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro",
                ))
        except Exception:
            results["auc"] = None

        return results


# ===================================================================
# ImageNet Pretrained Baseline
# ===================================================================

class ImageNetBaselineEvaluator:
    """
    Evaluate using an ImageNet-pretrained Vision Transformer (or ResNet)
    as a feature extractor, then linear-probe on top.

    This serves as the comparison *baseline*:
    "How much better is MedJEPA pre-training than generic ImageNet features?"
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "vit_b_16",
        # One of: "vit_b_16", "resnet50"
    ):
        self.device = get_device()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.backbone, self.embed_dim = self._build_backbone(backbone)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.probe = nn.Linear(self.embed_dim, num_classes).to(self.device)

    @staticmethod
    def _build_backbone(name: str):
        """
        Build an ImageNet-pretrained backbone. Returns (model, embed_dim).
        """
        try:
            import torchvision.models as tvm
        except ImportError:
            raise ImportError("torchvision is required for ImageNet baseline")

        if name == "vit_b_16":
            weights = tvm.ViT_B_16_Weights.DEFAULT
            model = tvm.vit_b_16(weights=weights)
            embed_dim = model.heads.head.in_features
            model.heads = nn.Identity()  # remove classification head
            return model, embed_dim
        elif name == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT
            model = tvm.resnet50(weights=weights)
            embed_dim = model.fc.in_features
            model.fc = nn.Identity()  # remove classification head
            return model, embed_dim
        else:
            raise ValueError(f"Unknown backbone: {name}. Use 'vit_b_16' or 'resnet50'.")

    def extract_features(self, dataloader: DataLoader):
        """Extract features from the ImageNet backbone."""
        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.backbone(images)
                if features.dim() == 3:
                    features = features.mean(dim=1)
                all_features.append(features.cpu())
                all_labels.append(labels)
        return torch.cat(all_features), torch.cat(all_labels)

    def train_probe(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
    ):
        """Train a linear probe on extracted ImageNet features."""
        dataset = TensorDataset(train_features, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.probe.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.probe.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for feats, labels in loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                logits = self.probe(feats)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"  ImageNet Probe Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {total_loss / len(loader):.4f}")

    def evaluate(
        self,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> dict:
        """Evaluate the ImageNet baseline probe."""
        self.probe.eval()
        with torch.no_grad():
            test_features = test_features.to(self.device)
            logits = self.probe(test_features)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        labels = test_labels.numpy()

        results = {
            "accuracy": float(accuracy_score(labels, preds)),
            "report": classification_report(labels, preds, output_dict=True),
        }

        try:
            if probs.shape[1] == 2:
                results["auc"] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                results["auc"] = float(roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro",
                ))
        except Exception:
            results["auc"] = None

        return results
