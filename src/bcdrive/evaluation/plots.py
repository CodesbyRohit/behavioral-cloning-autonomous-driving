"""Matplotlib figures for evaluation reports (headless, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRUE_C, PRED_C, GRID_C = "#5b6170", "#2f6fdb", "#e3e6eb"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)


def save_eval_plots(out_dir: Path, y_true, y_pred, metrics: dict, history: list[dict], title: str) -> None:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, color=PRED_C, linewidths=0)
    ax.plot([-1, 1], [-1, 1], color=TRUE_C, linewidth=1, linestyle="--", label="perfect")
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="recorded steering", ylabel="predicted steering",
           title=f"Prediction vs ground truth\nMSE {metrics['mse']:.4f}, MAE {metrics['mae']:.3f}")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_true.png", dpi=120)
    plt.close(fig)

    n = min(len(y_true), 600)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(y_true[:n], color=TRUE_C, linewidth=1.4, label="recorded (human)")
    ax.plot(y_pred[:n], color=PRED_C, linewidth=1.2, label="predicted (CNN)")
    ax.set(xlabel="held-out frame (recording order)", ylabel="steering", ylim=(-1.05, 1.05),
           title=f"Steering over the first {n} held-out frames — {title}")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "timeline.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(y_pred - y_true, bins=60, color=PRED_C, alpha=0.85)
    ax.axvline(0, color=TRUE_C, linewidth=1)
    ax.set(xlabel="residual (pred − true)", ylabel="frames", title="Residual distribution")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=120)
    plt.close(fig)

    bins = [b for b in metrics["by_bin"] if b["n"] > 0]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = [f"{b['range'][0]:+.2f}..{b['range'][1]:+.2f}" for b in bins]
    colors = [PRED_C if b["sufficient"] else "#b9c6e4" for b in bins]
    ax.bar(labels, [b["mae"] for b in bins], color=colors)
    for i, b in enumerate(bins):
        ax.text(i, b["mae"], f"n={b['n']}", ha="center", va="bottom", fontsize=7, color=TRUE_C)
    ax.set(xlabel="recorded steering bin", ylabel="MAE",
           title="Error by steering bin (pale = too few samples)")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "per_bin_mae.png", dpi=120)
    plt.close(fig)

    if history:
        save_loss_curves(out_dir / "loss_curves.png", history)


def save_loss_curves(path: Path, history: list[dict]) -> None:
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, [r.get("loss") for r in history], color=TRUE_C, marker="o", markersize=3,
            label="train (augmented, dropout on)")
    ax.plot(epochs, [r.get("val_loss") for r in history], color=PRED_C, marker="o", markersize=3,
            label="validation")
    ax.set(xlabel="epoch", ylabel="MSE", title="Training and validation loss")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
