"""
Tests for MedJEPA core components.

Run with:  python -m pytest tests/ -v
"""

import torch
import numpy as np
import pytest
import sys
import os

# Ensure medjepa is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestViTEncoder:
    def test_forward_shape(self):
        from medjepa.models.encoder import ViTEncoder

        encoder = ViTEncoder(
            image_size=224, patch_size=16, embed_dim=192, depth=2, num_heads=3
        )
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 196, 192)

    def test_forward_with_patch_indices(self):
        from medjepa.models.encoder import ViTEncoder

        encoder = ViTEncoder(
            image_size=224, patch_size=16, embed_dim=192, depth=2, num_heads=3
        )
        x = torch.randn(2, 3, 224, 224)
        indices = torch.tensor([0, 5, 10, 20, 50])
        out = encoder(x, patch_indices=indices)
        assert out.shape == (2, 5, 192)


class TestPatchEmbedding:
    def test_output_shape(self):
        from medjepa.models.encoder import PatchEmbedding

        pe = PatchEmbedding(image_size=224, patch_size=16, embed_dim=192)
        x = torch.randn(4, 3, 224, 224)
        out = pe(x)
        assert out.shape == (4, 196, 192)


# ---------------------------------------------------------------------------
# LeJEPA
# ---------------------------------------------------------------------------

class TestLeJEPA:
    @pytest.fixture
    def model(self):
        from medjepa.models.lejepa import LeJEPA

        return LeJEPA(
            image_size=64,
            patch_size=8,
            embed_dim=96,
            encoder_depth=2,
            encoder_heads=3,
            predictor_dim=48,
            predictor_depth=1,
            predictor_heads=3,
            mask_ratio=0.5,
        )

    def test_forward_returns_losses(self, model):
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert "total_loss" in out
        assert "prediction_loss" in out
        assert "regularization_loss" in out
        assert out["total_loss"].requires_grad

    def test_encode_returns_pooled(self, model):
        x = torch.randn(2, 3, 64, 64)
        emb = model.encode(x)
        assert emb.shape == (2, 96)
        assert not emb.requires_grad


# ---------------------------------------------------------------------------
# SIGReg Loss
# ---------------------------------------------------------------------------

class TestSIGRegLoss:
    def test_forward(self):
        from medjepa.training.losses import SIGRegLoss

        loss_fn = SIGRegLoss(lambda_reg=1.0)
        predicted = torch.randn(4, 10, 64)
        target = torch.randn(4, 10, 64)
        all_emb = torch.randn(4, 20, 64)
        out = loss_fn(predicted, target, all_emb)
        assert "total_loss" in out
        assert out["total_loss"].item() >= 0

    def test_prediction_loss_symmetric(self):
        from medjepa.training.losses import SIGRegLoss

        loss_fn = SIGRegLoss()
        a = torch.randn(4, 10, 64)
        b = torch.randn(4, 10, 64)
        # Identical inputs should give zero prediction loss
        loss_same = loss_fn.prediction_loss(a, a)
        assert loss_same.item() < 1e-5


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

class TestPatchMasker2D:
    def test_generate_mask(self):
        from medjepa.data.masking import PatchMasker2D

        masker = PatchMasker2D(image_size=224, patch_size=16, mask_ratio=0.75)
        ctx, tgt = masker.generate_mask()
        total = len(ctx) + len(tgt)
        assert total == 196  # 14*14

    def test_no_overlap(self):
        from medjepa.data.masking import PatchMasker2D

        masker = PatchMasker2D(image_size=64, patch_size=8, mask_ratio=0.5)
        ctx, tgt = masker.generate_mask()
        ctx_set = set(ctx.tolist())
        tgt_set = set(tgt.tolist())
        assert ctx_set.isdisjoint(tgt_set)


# ---------------------------------------------------------------------------
# Linear Probe
# ---------------------------------------------------------------------------

class TestLinearProbe:
    def test_train_and_evaluate(self):
        from medjepa.evaluation.linear_probe import LinearProbe, LinearProbeEvaluator

        # Tiny mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(1, 1)

            def encode(self, x):
                return torch.randn(x.shape[0], 64)

        model = MockModel()
        evaluator = LinearProbeEvaluator(model, num_classes=3, embed_dim=64)

        features = torch.randn(50, 64)
        labels = torch.randint(0, 3, (50,))
        evaluator.train_probe(features, labels, num_epochs=5)

        results = evaluator.evaluate(features, labels)
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Few-Shot
# ---------------------------------------------------------------------------

class TestFewShot:
    def test_data_efficiency(self):
        from medjepa.evaluation.few_shot import FewShotEvaluator

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(1, 1)

            def encode(self, x):
                return torch.randn(x.shape[0], 64)

        model = MockModel()
        ev = FewShotEvaluator(model, k=3)

        train_f = torch.randn(40, 64)
        train_l = torch.randint(0, 3, (40,))
        test_f = torch.randn(20, 64)
        test_l = torch.randint(0, 3, (20,))

        results = ev.evaluate_data_efficiency(
            train_f, train_l, test_f, test_l, fractions=[0.5, 1.0]
        )
        assert len(results) == 2
        assert "accuracy" in results[0]


# ---------------------------------------------------------------------------
# Segmentation Head
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_forward_shape(self):
        from medjepa.evaluation.segmentation import SimpleSegmentationHead

        head = SimpleSegmentationHead(
            embed_dim=96, num_classes=2, image_size=64, patch_size=8
        )
        patches = torch.randn(2, 64, 96)  # 8*8 = 64 patches
        out = head(patches)
        assert out.shape == (2, 2, 64, 64)

    def test_dice_score(self):
        from medjepa.evaluation.segmentation import dice_score

        pred = torch.ones(2, 1, 32, 32)
        tgt = torch.ones(2, 1, 32, 32)
        score = dice_score(pred, tgt)
        assert abs(score - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Visualization (non-interactive, no plt.show)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_extract_attention_weights(self, tmp_path):
        from medjepa.models.lejepa import LeJEPA
        from medjepa.utils.visualization import extract_attention_weights

        model = LeJEPA(
            image_size=64, patch_size=8, embed_dim=96,
            encoder_depth=2, encoder_heads=3,
            predictor_dim=48, predictor_depth=1, predictor_heads=3,
        )
        img = torch.randn(3, 64, 64)
        attn = extract_attention_weights(model, img)
        assert attn.shape == (8, 8)  # grid_size = 64/8 = 8
        assert attn.min() >= 0 and attn.max() <= 1.0 + 1e-6

    def test_plot_training_history(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from medjepa.utils.visualization import plot_training_history

        history = {"epochs": [0, 1, 2], "train_loss": [1.0, 0.5, 0.2]}
        save = str(tmp_path / "hist.png")
        plot_training_history(history, save_path=save)
        assert os.path.exists(save)

    def test_plot_evaluation_summary(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from medjepa.utils.visualization import plot_evaluation_summary

        results = {"Linear Probe": 0.82, "5-shot": 0.65}
        save = str(tmp_path / "summary.png")
        plot_evaluation_summary(results, save_path=save)
        assert os.path.exists(save)


# ---------------------------------------------------------------------------
# Device util
# ---------------------------------------------------------------------------

class TestDevice:
    def test_get_device(self):
        from medjepa.utils.device import get_device

        d = get_device()
        assert str(d) in ("cpu", "cuda", "mps")

    def test_get_device_info(self):
        from medjepa.utils.device import get_device_info

        # get_device_info() prints and returns None — just verify no crash
        get_device_info()


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_top_level(self):
        import medjepa
        assert hasattr(medjepa, "__version__")

    def test_models_package(self):
        from medjepa.models import ViTEncoder, LeJEPA, VJEPA

    def test_training_package(self):
        from medjepa.training import SIGRegLoss, MedJEPATrainer

    def test_evaluation_package(self):
        from medjepa.evaluation import (
            LinearProbe,
            LinearProbeEvaluator,
            FewShotEvaluator,
            SimpleSegmentationHead,
            dice_score,
        )

    def test_utils_package(self):
        from medjepa.utils import (
            get_device,
            get_device_info,
            plot_training_history,
            plot_embedding_space,
            plot_attention_map,
            extract_attention_weights,
        )
