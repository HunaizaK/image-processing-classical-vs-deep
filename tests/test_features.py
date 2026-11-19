import cv2
import numpy as np
import pytest

from signs import NUM_CLASSES
from signs.classical import SignFeatures, build_pipeline, equalize, hue_histogram
from signs.config import load_config
from signs.data import load_image, track_split
from signs.robustness import CORRUPTIONS, corrupt


def test_class_names_cover_all_43_classes():
    assert NUM_CLASSES == 43


@pytest.mark.parametrize(("kind", "dim"), [("pixels", 32 * 32), ("hog", 1764), ("hog_clahe", 1764), ("hog_clahe_color", 1764 + 16)])
def test_feature_dimensions(toy_images, kind, dim):
    """HOG on 32×32: 7×7 overlapping blocks × 2×2 cells × 9 bins = 1764 values."""
    x, _ = toy_images
    assert SignFeatures(kind).transform(x[:2]).shape == (2, dim)


def test_hue_histogram_separates_red_from_blue(toy_images):
    x, y = toy_images
    red, blue = hue_histogram(x[y == 0][0], 16), hue_histogram(x[y == 1][0], 16)
    assert red.argmax() != blue.argmax()
    assert red.sum() == pytest.approx(1, abs=1e-4)


def test_clahe_brightens_dark_image_and_keeps_shape():
    ramp = np.tile(np.linspace(10, 40, 32, dtype=np.uint8), (32, 1))  # dim, low-contrast gradient
    dark = np.repeat(ramp[..., None], 3, axis=2)
    out = equalize(dark, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)))
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).std() > cv2.cvtColor(dark, cv2.COLOR_RGB2GRAY).std()


def test_classical_pipeline_learns_toy_signs(toy_images):
    x, y = toy_images
    model = build_pipeline("hog_clahe_color", 0.1, load_config()).fit(x[::2], y[::2])
    assert (model.predict(x[1::2]) == y[1::2]).mean() > 0.9  # 60 training images, 1780 features


def test_track_split_never_shares_a_track():
    groups = np.repeat([f"t{i}" for i in range(20)], 30)
    tr, va = track_split(groups, 0.2, seed=0)
    assert not set(groups[tr]) & set(groups[va])
    assert len(set(groups[va])) == 4


def test_load_image_crops_roi_and_resizes(tmp_path):
    img = np.zeros((40, 50, 3), np.uint8)
    img[10:30, 20:40] = (0, 0, 255)  # BGR red patch = the ROI
    path = tmp_path / "x.ppm"
    cv2.imwrite(str(path), img)
    out = load_image(path, (20, 10, 39, 29), 32)
    assert out.shape == (32, 32, 3)
    assert (out[..., 0] == 255).all() and (out[..., 2] == 0).all()  # RGB order, only the patch remains


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruptions_keep_shape_dtype_and_get_stronger(toy_images, name):
    x, _ = toy_images
    clean = x[:8].astype(np.float32)
    diffs = []
    for s in (1, 3):
        out = corrupt(x[:8], name, s, seed=0)
        assert out.shape == x[:8].shape and out.dtype == np.uint8
        diffs.append(np.abs(out.astype(np.float32) - clean).mean())
    assert diffs[1] > diffs[0] > 0
