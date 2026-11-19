"""Metrics, significance tests, error analysis and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from signs import CLASS_NAMES, NUM_CLASSES  # noqa: E402


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on paired predictions: only articles where the models disagree count."""
    a, b = pred_a == y, pred_b == y
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    correct = (y == pred).astype(float)
    samples = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)]


def per_class_accuracy(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES))
    return cm.diagonal() / np.maximum(cm.sum(1), 1)


def worst_classes(y: np.ndarray, pred: np.ndarray, k: int = 6) -> list[dict]:
    acc = per_class_accuracy(y, pred)
    present = [i for i in np.argsort(acc) if (y == i).any()]  # classes absent from y are not "0% accurate"
    return [{"class": CLASS_NAMES[i], "accuracy": round(float(acc[i]), 4), "n": int((y == i).sum())} for i in present[:k]]


def confusion_pairs(y: np.ndarray, pred: np.ndarray, k: int = 6) -> list[dict]:
    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    return [{"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "count": int(cm[i, j])} for i, j in zip(*np.unravel_index(flat, cm.shape), strict=True)]


def accuracy_by_size(y: np.ndarray, preds: dict[str, np.ndarray], side: np.ndarray) -> list[dict]:
    """Original sign size in pixels (before resizing): small = far away = few real pixels."""
    edges = [0, 20, 30, 45, 1000]
    names = ["<20 px", "20-29 px", "30-44 px", ">=45 px"]
    rows = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names, strict=True):
        m = (side >= lo) & (side < hi)
        rows.append({"size": name, "n": int(m.sum()), **{k: round(float((p[m] == y[m]).mean()), 4) for k, p in preds.items()}})
    return rows


def plot_examples(x: np.ndarray, y: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(3, 12, figsize=(12, 3.4))
    for ax, cls in zip(axes.ravel(), range(36), strict=True):
        ax.imshow(x[np.flatnonzero(y == cls)[0]])
        ax.set_title(str(cls), fontsize=7)
        ax.axis("off")
    fig.suptitle("One training image per class (after crop + resize to 32×32)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_preprocessing(img: np.ndarray, equalized: np.ndarray, path: Path, cell: int = 4) -> None:
    """Raw → CLAHE → gradient magnitude → dominant HOG orientation per cell."""
    gray = cv2.cvtColor(equalized, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    axes[0].imshow(img)
    axes[0].set_title("Raw (dark)")
    axes[1].imshow(equalized)
    axes[1].set_title("CLAHE")
    axes[2].imshow(mag, cmap="gray")
    axes[2].set_title("Gradient magnitude")
    axes[3].imshow(np.zeros_like(gray), cmap="gray")
    for cy in range(0, gray.shape[0], cell):
        for cx in range(0, gray.shape[1], cell):
            m = mag[cy : cy + cell, cx : cx + cell]
            a = np.deg2rad(ang[cy : cy + cell, cx : cx + cell].ravel()[m.argmax()] + 90)  # edge runs perpendicular to the gradient
            length = cell / 2 * min(1.0, m.max() / (mag.max() / 3 + 1e-6))
            c = (cx + cell / 2, cy + cell / 2)
            axes[3].plot([c[0] - length * np.cos(a), c[0] + length * np.cos(a)], [c[1] - length * np.sin(a), c[1] + length * np.sin(a)], "w-", lw=1.2)
    axes[3].set_title("Dominant edge per HOG cell")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(x: np.ndarray, y: np.ndarray, pred: np.ndarray, conf: np.ndarray, title: str, path: Path, n: int = 16) -> None:
    wrong = np.flatnonzero(pred != y)
    wrong = wrong[np.argsort(conf[wrong])[::-1]][:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.95, 5.4))
    for ax, i in zip(axes.ravel(), wrong, strict=False):
        ax.imshow(x[i])
        ax.set_title(f"true: {CLASS_NAMES[y[i]][:16]}\npred: {CLASS_NAMES[pred[i]][:16]}\n{conf[i]:.2f}", fontsize=6.5)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_robustness(rows: list[dict], clean: dict[str, float], path: Path) -> None:
    names = sorted({r["corruption"] for r in rows})
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharey=True)
    for ax, name in zip(axes, names, strict=True):
        for model, style in (("classical", "o--"), ("cnn", "s-")):
            pts = [(0, clean[model])] + [(r["severity"], r[model]) for r in rows if r["corruption"] == name]
            ax.plot(*zip(*pts, strict=True), style, label=model)
        ax.set(title=name.replace("_", " "), xlabel="severity", xticks=[0, 1, 2, 3])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_corruption_examples(img: np.ndarray, corrupt_fn, names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(len(names), 4, figsize=(4.4, 1.15 * len(names)))
    for r, name in enumerate(names):
        for s in range(4):
            axes[r, s].imshow(img if s == 0 else corrupt_fn(img[None], name, s)[0])
            axes[r, s].axis("off")
            if r == 0:
                axes[r, s].set_title("clean" if s == 0 else f"sev {s}", fontsize=8)
        axes[r, 0].text(-4, 16, name.replace("_", " "), ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history["accuracy"], label="train (augmented)")
    ax.plot(history["val_accuracy"], label="validation")
    ax.set(xlabel="epoch", ylabel="accuracy", title="CNN training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
