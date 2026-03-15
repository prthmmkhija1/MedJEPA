"""
Minimal MedJEPA — the core JEPA loop in ~80 lines.

This script demonstrates the complete self-supervised training idea
without any engineering overhead: no checkpointing, no mixed precision,
no EMA. Just the pure architecture.

Usage:
    python examples/minimal_medjepa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. Encoder: ViT that converts image patches to embeddings ----------

class TinyViT(nn.Module):
    def __init__(self, image_size=64, patch_size=8, embed_dim=128, depth=4, heads=4):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, indices=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        if indices is not None:
            x = x[:, indices]
        return self.norm(self.blocks(x))

# ---------- 2. Predictor: predicts target embeddings from context ----------

class Predictor(nn.Module):
    def __init__(self, embed_dim=128, pred_dim=64, num_patches=64, depth=2, heads=4):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, pred_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(pred_dim, heads, pred_dim * 4,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.proj_out = nn.Linear(pred_dim, embed_dim)

    def forward(self, ctx_emb, ctx_idx, tgt_idx):
        B = ctx_emb.shape[0]
        ctx = self.proj_in(ctx_emb) + self.pos_embed[:, ctx_idx]
        masks = self.mask_token.expand(B, len(tgt_idx), -1) + self.pos_embed[:, tgt_idx]
        out = self.blocks(torch.cat([ctx, masks], dim=1))
        return self.proj_out(out[:, -len(tgt_idx):])

# ---------- 3. SIGReg Loss: prediction + variance + covariance ----------

def sigreg_loss(predicted, target, all_emb, lam=1.0):
    pred_loss = F.mse_loss(predicted, target)                   # prediction accuracy
    std = all_emb.std(dim=0)                                    # (T, D)
    var_loss = F.softplus(1.0 - std, beta=5.0).mean()           # anti-collapse
    centered = all_emb.mean(dim=1).float() - all_emb.mean(dim=1).mean(dim=0)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    off_diag = cov - torch.diag(cov.diagonal())
    cov_loss = (off_diag ** 2).sum() / cov.shape[0]             # decorrelation
    return pred_loss + lam * (2.5 * var_loss + 0.04 * cov_loss)

# ---------- 4. Training loop ----------

def main():
    IMG_SIZE, PATCH_SIZE, EMBED_DIM = 64, 8, 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2          # 64 patches
    MASK_RATIO = 0.75

    encoder = TinyViT(IMG_SIZE, PATCH_SIZE, EMBED_DIM, depth=4, heads=4)
    predictor = Predictor(EMBED_DIM, 64, NUM_PATCHES, depth=2, heads=4)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=3e-4
    )

    # Simulate training on random images (replace with your DataLoader)
    for step in range(200):
        images = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)     # fake batch

        # Mask: randomly choose context (25%) and target (75%) patches
        perm = torch.randperm(NUM_PATCHES)
        n_ctx = NUM_PATCHES - int(NUM_PATCHES * MASK_RATIO)
        ctx_idx, tgt_idx = perm[:n_ctx].sort().values, perm[n_ctx:].sort().values

        # Encode context (with gradient) and target (no gradient)
        ctx_emb = encoder(images, ctx_idx)
        with torch.no_grad():
            tgt_emb = encoder(images, tgt_idx)

        # Predict and compute loss
        predicted = predictor(ctx_emb, ctx_idx, tgt_idx)
        loss = sigreg_loss(predicted, tgt_emb.detach(), ctx_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:3d}  loss={loss.item():.4f}")

    print("Done! Encoder can now be used for downstream tasks.")
    emb = encoder(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))  # (1, 64, 128)
    print(f"Representation shape: {emb.mean(dim=1).shape}")  # (1, 128)

if __name__ == "__main__":
    main()
