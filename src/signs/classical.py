"""Classical pipeline: OpenCV preprocessing + hand-crafted features (HOG, colour) + linear SVM.

Everything, including preprocessing, lives inside one sklearn Pipeline that takes raw RGB
uint8 images, so the saved model can be served without re-implementing feature code.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from signs.config import Config, HogConfig

log = logging.getLogger(__name__)


def equalize(img: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """CLAHE on the lightness channel only: fixes dark/backlit signs without distorting colour."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def make_hog(size: int, hcfg: HogConfig) -> cv2.HOGDescriptor:
    cell = hcfg.cell
    block = cell * hcfg.block
    # (window, block, block stride, cell, bins). Blocks overlap by one cell; each block is L2-Hys normalised,
    # which makes the descriptor robust to local illumination changes.
    return cv2.HOGDescriptor((size, size), (block, block), (cell, cell), (cell, cell), hcfg.bins)


def hue_histogram(img: np.ndarray, bins: int) -> np.ndarray:
    """Hue histogram weighted by saturation: red/blue/yellow sign colour, ignoring grey background."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = np.histogram(hsv[..., 0], bins=bins, range=(0, 180), weights=hsv[..., 1].astype(np.float32))[0]
    return hist / (hist.sum() + 1e-6)


class SignFeatures(BaseEstimator, TransformerMixin):
    """kind: 'pixels' | 'hog' | 'hog_clahe' | 'hog_clahe_color'"""

    def __init__(self, kind: str = "hog_clahe_color", size: int = 32, cell: int = 4, block: int = 2, bins: int = 9, color_bins: int = 16):
        self.kind = kind
        self.size = size
        self.cell = cell
        self.block = block
        self.bins = bins
        self.color_bins = color_bins

    def fit(self, x, y=None):
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "pixels":
            return np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).ravel() / 255.0 for img in x]).astype(np.float32)
        hog = make_hog(self.size, HogConfig(cell=self.cell, block=self.block, bins=self.bins))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        feats = []
        for img in x:
            if self.kind != "hog":
                img = equalize(img, clahe)
            f = hog.compute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)).ravel()
            if self.kind == "hog_clahe_color":
                f = np.concatenate([f, hue_histogram(img, self.color_bins)])
            feats.append(f)
        return np.asarray(feats, dtype=np.float32)


def build_pipeline(kind: str, c: float, cfg: Config) -> Pipeline:
    h = cfg.classical.hog
    features = SignFeatures(kind, cfg.data.image_size, h.cell, h.block, h.bins, cfg.classical.color_bins)
    return Pipeline([("features", features), ("scale", StandardScaler()), ("svm", LinearSVC(C=c, random_state=cfg.seed))])


def ablation(ds, cfg: Config) -> tuple[dict, str, float]:
    """Validation accuracy for every feature set × C. Returns results and the best (kind, C)."""
    results: dict[str, dict[float, float]] = {}
    for kind in ["pixels", "hog", "hog_clahe", "hog_clahe_color"]:
        # Precompute features once per kind, then sweep C on the cached matrix.
        feats = build_pipeline(kind, 1.0, cfg).named_steps["features"]
        f_tr, f_va = feats.transform(ds.x_train), feats.transform(ds.x_val)
        scaler = StandardScaler().fit(f_tr)
        f_tr, f_va = scaler.transform(f_tr), scaler.transform(f_va)
        results[kind] = {}
        for c in cfg.classical.c_grid:
            svm = LinearSVC(C=c, random_state=cfg.seed).fit(f_tr, ds.y_train)
            results[kind][c] = round(float(accuracy_score(ds.y_val, svm.predict(f_va))), 4)
            log.info("%-16s C=%-5g val acc=%.4f (features=%d)", kind, c, results[kind][c], f_tr.shape[1])
    best_kind = max(results, key=lambda k: max(results[k].values()))
    best_c = max(results[best_kind], key=results[best_kind].get)
    return results, best_kind, best_c
