"""Image corruptions built with OpenCV, used to stress-test both models.

Each corruption takes an RGB uint8 image and a severity 1..3 and returns a uint8 image.
They model real camera failure modes: motion, focus, sensor noise, low light, tilt, occlusion.
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_noise(img: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, [8, 16, 28][s - 1], img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def defocus_blur(img: np.ndarray, s: int, rng=None) -> np.ndarray:
    k = [3, 5, 7][s - 1]
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img: np.ndarray, s: int, rng=None) -> np.ndarray:
    k = [3, 5, 7][s - 1]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k  # horizontal streak, like a car passing the sign
    return cv2.filter2D(img, -1, kernel)


def darken(img: np.ndarray, s: int, rng=None) -> np.ndarray:
    """Gamma > 1 darkens mid-tones more than highlights: closer to dusk than a plain scale."""
    gamma = [1.8, 2.6, 3.6][s - 1]
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def rotate(img: np.ndarray, s: int, rng=None) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), [10, 20, 30][s - 1], 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def occlude(img: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Grey square at a random position (a branch, a sticker, another car)."""
    out = img.copy()
    h, w = img.shape[:2]
    size = int(w * [0.25, 0.35, 0.45][s - 1])
    y, x = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
    out[y : y + size, x : x + size] = 128
    return out


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "darken": darken,
    "rotate": rotate,
    "occlude": occlude,
}


def corrupt(images: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fn = CORRUPTIONS[name]
    return np.stack([fn(img, severity, rng) for img in images])
