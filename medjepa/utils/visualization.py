"""
Visualization tools for understanding what MedJEPA learned.

These visuals help answer:
- What parts of the image does the model pay attention to?
- Do similar diseases have similar embeddings?
- Can we see what features the model learned?
"""

import torch
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
