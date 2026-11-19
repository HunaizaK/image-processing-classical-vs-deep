"""Typed configuration loaded from YAML, with an optional `full:` override block."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_dir: Path
    cache: Path
    base_url: str
    files: dict[str, str]
    image_size: int
    val_fraction: float
    train_limit: int | None = None


class HogConfig(BaseModel):
    cell: int
    block: int
    bins: int


class ClassicalConfig(BaseModel):
    hog: HogConfig
    color_bins: int
    c_grid: list[float]


class CnnConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    dropout: float
    patience: int


class RobustnessConfig(BaseModel):
    severities: list[int]


class Config(BaseModel):
    seed: int
    data: DataConfig
    classical: ClassicalConfig
    cnn: CnnConfig
    robustness: RobustnessConfig
    artifacts_dir: Path
    reports_dir: Path


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: str | Path = "configs/config.yaml", full: bool = False) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    overrides = raw.pop("full", {}) or {}
    if full:
        raw = _merge(raw, overrides)
    return Config(**raw)
