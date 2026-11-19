"""Download, verify, crop, resize and split GTSRB.

GTSRB's training set is recorded as *tracks*: ~30 consecutive video frames of the same physical
sign. Frames in a track are near-duplicates, so validation is split by track (GroupShuffleSplit);
a random frame-level split would leak and overstate accuracy.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from signs.config import Config

log = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(cfg: Config) -> Path:
    raw = Path(cfg.data.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name, expected in cfg.data.files.items():
        path = raw / name
        if not path.exists():
            log.info("Downloading %s", name)
            urllib.request.urlretrieve(f"{cfg.data.base_url}/{name}", path)
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch for {path}")
        marker = raw / f".{name}.extracted"
        if not marker.exists():
            with zipfile.ZipFile(path) as zf:
                zf.extractall(raw)
            marker.touch()
    return raw


def load_image(path: Path, roi: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Read a .ppm, crop to the annotated sign region, resize to size×size RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    x1, y1, x2, y2 = roi
    img = img[y1 : y2 + 1, x1 : x2 + 1]
    # INTER_AREA averages source pixels when shrinking (anti-aliasing); most signs are larger than 32 px.
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_annotations(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = raw / "GTSRB" / "Final_Training" / "Images"
    frames = []
    for csv in sorted(train_dir.glob("*/GT-*.csv")):
        df = pd.read_csv(csv, sep=";")
        df["path"] = [str(csv.parent / f) for f in df["Filename"]]
        # Filename = <track>_<frame>.ppm; a track id is unique only within its class.
        df["group"] = df["ClassId"].astype(str) + "_" + df["Filename"].str.split("_").str[0]
        frames.append(df)
    train = pd.concat(frames, ignore_index=True)
    test = pd.read_csv(raw / "GT-final_test.csv", sep=";")
    test["path"] = [str(raw / "GTSRB" / "Final_Test" / "Images" / f) for f in test["Filename"]]
    test["group"] = "test"
    return train, test


def to_arrays(df: pd.DataFrame, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].to_numpy()
    images = np.stack([load_image(Path(p), tuple(r), size) for p, r in zip(df["path"], rois, strict=True)])
    # Original sign size (min side of the ROI) is kept for the "small, distant signs" error analysis.
    side = np.minimum(rois[:, 2] - rois[:, 0], rois[:, 3] - rois[:, 1]) + 1
    return images, df["ClassId"].to_numpy(), side


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    side_test: np.ndarray


def track_split(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, val_idx


def build_cache(cfg: Config) -> Path:
    cache = Path(cfg.data.cache)
    if cache.exists():
        return cache
    raw = download_and_extract(cfg)
    train, test = read_annotations(raw)
    log.info("Decoding %d training and %d test images", len(train), len(test))
    x_tr, y_tr, _ = to_arrays(train, cfg.data.image_size)
    x_te, y_te, side_te = to_arrays(test, cfg.data.image_size)
    np.savez_compressed(cache, x_train=x_tr, y_train=y_tr, g_train=train["group"].to_numpy(), x_test=x_te, y_test=y_te, side_test=side_te)
    return cache


def load_dataset(cfg: Config) -> Dataset:
    d = np.load(build_cache(cfg), allow_pickle=True)
    x, y, g = d["x_train"], d["y_train"], d["g_train"]
    if cfg.data.train_limit:
        rng = np.random.default_rng(cfg.seed)
        keep = np.sort(rng.choice(len(x), cfg.data.train_limit, replace=False))
        x, y, g = x[keep], y[keep], g[keep]
    tr, va = track_split(g, cfg.data.val_fraction, cfg.seed)
    ds = Dataset(x[tr], y[tr], g[tr], x[va], y[va], d["x_test"], d["y_test"], d["side_test"])
    log.info("train=%d val=%d test=%d (val split by %d tracks)", len(tr), len(va), len(ds.y_test), len(set(g[va])))
    return ds
