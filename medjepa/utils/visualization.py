"""
Visualization tools for understanding what MedJEPA learned.

These visuals help answer:
- What parts of the image does the model pay attention to?
- Do similar diseases have similar embeddings?
- Can we see what features the model learned?
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot loss curves from training."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(history["epochs"], history["train_loss"], label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(history["epochs"], history["val_loss"], label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MedJEPA Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Embedding Space (t-SNE)",
    save_path: Optional[str] = None,
):
    """
    Visualize the learned embedding space using t-SNE.

    t-SNE takes high-dimensional embeddings (768 numbers per image)
    and projects them to 2D so we can see them.

    What to look for:
    - Images of the same disease should cluster together
    - Different diseases should be separated
    """
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[i] if class_names else f"Class {label}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.6,
            s=10,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_map(
    image: np.ndarray,
    attention_weights: np.ndarray,
    title: str = "Attention Map",
    save_path: Optional[str] = None,
):
    """
    Overlay attention weights on a medical image.

    Shows WHERE the model is looking when analyzing the image.
    Bright areas = model pays more attention there.

    Clinicians can use this to verify the model is looking at
    the right areas (e.g., the tumor, not the text annotation).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention map
    axes[1].imshow(attention_weights, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_weights, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_data_efficiency(
    results: list,
    baseline_accuracy: float = None,
    title: str = "Data Efficiency Curve",
    save_path: Optional[str] = None,
):
    """
    Plot how accuracy changes with amount of labeled data.

    This is the MONEY PLOT for self-supervised learning:
    Shows that MedJEPA needs far fewer labels than supervised methods.
    """
    fractions = [r["fraction"] * 100 for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(fractions, accuracies, "bo-", linewidth=2, markersize=8, label="MedJEPA")

    if baseline_accuracy:
        ax.axhline(
            y=baseline_accuracy * 100, color="r", linestyle="--",
            label=f"Supervised baseline ({baseline_accuracy*100:.1f}%)"
        )

    ax.set_xlabel("% of Labeled Training Data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def extract_attention_weights(model, image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract attention weights from the last transformer block.

    Works with nn.MultiheadAttention (our TransformerBlock uses
    ``self.attention = nn.MultiheadAttention(...)``).

    Returns a 2D attention map (H_patches x W_patches) that can be
    resized to the original image dimensions for overlay.
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        # Get patch embeddings from encoder
        encoder = model.encoder if hasattr(model, "encoder") else model

        # Forward through patch embedding
        x = encoder.patch_embed(image_tensor)
        x = x + encoder.pos_embed

        # Forward through all transformer blocks, capture last attention
        attn_weights = None
        for block in encoder.blocks:
            normed = block.norm1(x)
            # nn.MultiheadAttention returns (output, attn_weights)
            # when need_weights=True and average_attn_weights=False
            # we get shape (B, num_heads, N, N)
            _, aw = block.attention(
                normed, normed, normed,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_weights = aw  # keep the last block's weights

            # Continue the full forward pass so x is correct for next block
            x = block(x)

    if attn_weights is None:
        raise ValueError("Could not extract attention weights from model.")

    # attn_weights shape: (B, num_heads, N, N) — average across heads
    attn_map = attn_weights.mean(dim=1)[0]  # (N, N)

    # Use mean attention from all patches (self-attention rollout approximation)
    attn_map = attn_map.mean(dim=0)  # (N,)

    # Reshape to 2D grid
    num_patches = attn_map.shape[0]
    grid_size = int(num_patches ** 0.5)
    attn_map_2d = attn_map.reshape(grid_size, grid_size).cpu().numpy()

    # Normalize to [0, 1]
    attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (
        attn_map_2d.max() - attn_map_2d.min() + 1e-8
    )

    return attn_map_2d


def plot_reconstruction_comparison(
    original: np.ndarray,
    context_patches: np.ndarray,
    predicted_patches: np.ndarray,
    title: str = "JEPA Prediction vs Original",
    save_path: Optional[str] = None,
):
    """
    Show what the JEPA predictor reconstructs in latent space,
    projected back to image space for visualization.

    Columns: Original | Context (visible) | Predicted (target)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(context_patches)
    axes[1].set_title("Context (visible patches)")
    axes[1].axis("off")

    axes[2].imshow(predicted_patches)
    axes[2].set_title("Predicted regions")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_evaluation_summary(
    eval_results: dict,
    title: str = "MedJEPA Evaluation Summary",
    save_path: Optional[str] = None,
):
    """
    Create a bar chart summarizing evaluation metrics.

    Args:
        eval_results: Dict with metric names as keys, values as floats (0-1).
    """
    metrics = list(eval_results.keys())
    values = [eval_results[m] * 100 for m in metrics]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bars = ax.bar(metrics, values, color=plt.cm.Set2(np.linspace(0, 1, len(metrics))))

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ===================================================================
# GradCAM — Gradient-weighted Class Activation Mapping
# ===================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Vision Transformers.

    GradCAM highlights the image regions most responsible for a particular
    class prediction by combining the gradients flowing back into the last
    transformer block with its activations.

    Usage::

        cam = GradCAM(model)
        heatmap = cam(image_tensor, target_class=3)
        # heatmap is a (H_grid, W_grid) numpy array in [0, 1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """
        Args:
            model: A LeJEPA (or any model with ``.encoder.blocks``).
            target_layer: Name of the layer to hook. If None, uses the last
                          transformer block's ``norm1`` output (pre-attention).
        """
        self.model = model
        self.device = next(model.parameters()).device

        # Resolve target layer
        encoder = model.encoder if hasattr(model, "encoder") else model
        if target_layer is not None:
            self._target = dict(encoder.named_modules())[target_layer]
        else:
            # Default: last block's first norm (captures token activations)
            self._target = encoder.blocks[-1].norm1

        # Storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_handle = self._target.register_forward_hook(self._save_activation)
        self._bwd_handle = self._target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        probe: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Args:
            image: (C, H, W) or (1, C, H, W) input tensor.
            target_class: Class index to compute the CAM for.  If None,
                          uses the predicted class.
            probe: A classification head (e.g. ``LinearProbe``) appended on
                   top of the encoder.  If None, the raw CLS / mean-token
                   logits are used.

        Returns:
            heatmap: 2D numpy array (grid_H, grid_W) in [0, 1].
        """
        self.model.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device).requires_grad_(True)

        # Forward through encoder to get token embeddings
        encoder = self.model.encoder if hasattr(self.model, "encoder") else self.model
        tokens = encoder(image)  # (1, N, D)

        # Classification logits
        if probe is not None:
            probe = probe.to(self.device)
            pooled = tokens.mean(dim=1)  # (1, D)
            logits = probe(pooled)  # (1, C)
        else:
            logits = tokens.mean(dim=1)  # (1, D) – treat each dim as a "logit"

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        # Backward from target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not fire. Check target layer.")

        # Grad-weighted activations
        weights = self._gradients.mean(dim=-1, keepdim=True)  # (1, N, 1)
        cam = (weights * self._activations).sum(dim=-1)  # (1, N)
        cam = F.relu(cam)[0]  # (N,)

        # Reshape to 2D grid
        grid_size = int(cam.shape[0] ** 0.5)
        if grid_size * grid_size != cam.shape[0]:
            grid_size = int(np.ceil(cam.shape[0] ** 0.5))
            padded = torch.zeros(grid_size * grid_size, device=cam.device)
            padded[: cam.shape[0]] = cam
            cam = padded

        cam_2d = cam.reshape(grid_size, grid_size).cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_2d.min(), cam_2d.max()
        if cam_max - cam_min > 0:
            cam_2d = (cam_2d - cam_min) / (cam_max - cam_min)

        return cam_2d

    def remove_hooks(self):
        """Remove forward / backward hooks (call when done)."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def compute_saliency_map(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> np.ndarray:
    """
    Compute a vanilla gradient saliency map.

    Returns the absolute gradient of the model output w.r.t. the input
    image, averaged across colour channels.

    Args:
        model: LeJEPA or encoder model.
        image: (C, H, W) or (1, C, H, W) tensor.

    Returns:
        saliency: (H, W) numpy array normalised to [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device).requires_grad_(True)

    encoder = model.encoder if hasattr(model, "encoder") else model
    tokens = encoder(image)
    # Use squared norm of mean-pooled representation as scalar objective
    pooled = tokens.mean(dim=1)
    scalar = (pooled ** 2).sum()
    scalar.backward()

    grad = image.grad.data.abs()  # (1, C, H, W)
    saliency = grad[0].mean(dim=0).cpu().numpy()  # (H, W)

    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 0:
        saliency = (saliency - s_min) / (s_max - s_min)
    return saliency


def plot_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "GradCAM",
    save_path: Optional[str] = None,
):
    """
    Overlay a GradCAM heatmap on a medical image.

    Args:
        image: Original image (H, W) or (H, W, 3) in [0, 1].
        heatmap: (grid_H, grid_W) in [0, 1] — will be up-sampled.
        title: Plot title.
        save_path: If set, saves the figure.
    """
    from PIL import Image as PILImage

    if image.ndim == 3:
        display_img = image
    else:
        display_img = np.stack([image] * 3, axis=-1)

    H, W = display_img.shape[:2]

    # Up-sample heatmap to image resolution
    hm_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    hm_pil = hm_pil.resize((W, H), PILImage.BILINEAR)
    hm_up = np.array(hm_pil, dtype=np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(hm_up, cmap="jet")
    axes[1].set_title("GradCAM")
    axes[1].axis("off")

    axes[2].imshow(display_img)
    axes[2].imshow(hm_up, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
