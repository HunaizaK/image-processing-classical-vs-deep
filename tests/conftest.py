"""Synthetic 'signs' (coloured shapes) so tests run offline in seconds."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def draw_sign(kind: int, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), rng.integers(90, 160), np.uint8)
    c = size // 2 + int(rng.integers(-2, 3))
    if kind == 0:  # red circle
        cv2.circle(img, (c, c), 11, (220, 30, 30), -1)
    elif kind == 1:  # blue circle with arrow
        cv2.circle(img, (c, c), 11, (30, 60, 200), -1)
        cv2.arrowedLine(img, (c - 6, c), (c + 6, c), (255, 255, 255), 2)
    elif kind == 2:  # triangle
        pts = np.array([[c, c - 11], [c - 11, c + 9], [c + 11, c + 9]])
        cv2.fillPoly(img, [pts], (230, 230, 230))
        cv2.polylines(img, [pts], True, (220, 30, 30), 2)
    return img


@pytest.fixture(scope="session")
def toy_images() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 40)
    x = np.stack([draw_sign(k, rng) for k in y])
    return x, y
