import os
import numpy as np
import matplotlib.pyplot as plt


def plot_attention(attn, ed_idx, es_idx, pred_ed_idx=None, pred_es_idx=None, save_path=None):
    """Plot temporal attention with ground-truth and optional predicted ED/ES markers."""
    attn = np.asarray(attn)
    frames = np.arange(len(attn))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, attn, color="tab:blue", linewidth=2.0, label="Attention")

    ax.axvline(ed_idx, color="green", linestyle="-", linewidth=1.8, label=f"GT ED ({ed_idx})")
    ax.axvline(es_idx, color="red", linestyle="-", linewidth=1.8, label=f"GT ES ({es_idx})")

    if pred_ed_idx is not None:
        ax.axvline(
            pred_ed_idx,
            color="green",
            linestyle="--",
            linewidth=1.8,
            label=f"Pred ED ({pred_ed_idx})"
        )
    if pred_es_idx is not None:
        ax.axvline(
            pred_es_idx,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=f"Pred ES ({pred_es_idx})"
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Attention weight")
    ax.set_title("Temporal Attention")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_phase_curves(phase_probs, ed_idx, es_idx, pred_ed_idx, pred_es_idx, save_path=None):
    """
    Plot per-frame ED/ES probabilities and mark GT/predicted frame indices.
    phase_probs shape must be (T, 3): [background, ED, ES].
    """
    phase_probs = np.asarray(phase_probs)
    if phase_probs.ndim != 2 or phase_probs.shape[1] != 3:
        raise ValueError("phase_probs must have shape (T, 3)")

    frames = np.arange(phase_probs.shape[0])
    ed_curve = phase_probs[:, 1]
    es_curve = phase_probs[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(frames, ed_curve, color="green", linewidth=2.0, label="ED probability")
    axes[0].axvline(ed_idx, color="green", linestyle="-", linewidth=1.6, label=f"GT ED ({ed_idx})")
    axes[0].axvline(pred_ed_idx, color="green", linestyle="--", linewidth=1.6, label=f"Pred ED ({pred_ed_idx})")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("ED Phase Curve")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(frames, es_curve, color="red", linewidth=2.0, label="ES probability")
    axes[1].axvline(es_idx, color="red", linestyle="-", linewidth=1.6, label=f"GT ES ({es_idx})")
    axes[1].axvline(pred_es_idx, color="red", linestyle="--", linewidth=1.6, label=f"Pred ES ({pred_es_idx})")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("ES Phase Curve")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
