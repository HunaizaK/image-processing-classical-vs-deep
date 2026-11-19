"""FastAPI service: upload a cropped traffic-sign image, get the class.

SIGNS_BACKEND=cnn (default) or classical. Run: uvicorn signs.serve:app --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from signs import CLASS_NAMES

ARTIFACTS = Path(os.environ.get("SIGNS_ARTIFACTS", "artifacts"))
IMAGE_SIZE = 32
MAX_BYTES = 5_000_000


class Prediction(BaseModel):
    backend: str
    class_id: int
    label: str
    confidence: float
    top3: list[dict]


def decode(data: bytes, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bytes (PNG/JPEG/PPM) -> RGB uint8 size×size, resized exactly as in training."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a decodable image")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Predictor:
    def __init__(self, backend: str, artifacts: Path):
        self.backend = backend
        if backend == "classical":
            import joblib

            model = joblib.load(artifacts / "classical.joblib")

            def proba(x):
                s = model.decision_function(x)
                e = np.exp(s - s.max(1, keepdims=True))  # softmax over SVM margins: a ranking score, not calibrated
                return e / e.sum(1, keepdims=True)

            self._proba = proba
        elif backend == "cnn":
            import keras

            model = keras.models.load_model(artifacts / "cnn.keras")
            self._proba = lambda x: np.asarray(model(x, training=False))
        else:
            raise ValueError(f"Unknown backend {backend!r}")

    def predict(self, image: np.ndarray) -> Prediction:
        p = self._proba(image[None])[0]
        top = np.argsort(p)[::-1][:3]
        return Prediction(
            backend=self.backend,
            class_id=int(top[0]),
            label=CLASS_NAMES[top[0]],
            confidence=round(float(p[top[0]]), 4),
            top3=[{"label": CLASS_NAMES[i], "score": round(float(p[i]), 4)} for i in top],
        )


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(os.environ.get("SIGNS_BACKEND", "cnn"), ARTIFACTS)


app = FastAPI(title="Traffic sign classifier", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": get_predictor().backend}


@app.post("/predict", response_model=Prediction)
async def predict(file: Annotated[UploadFile, File()]) -> Prediction:
    data = await file.read()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(status_code=422, detail="Empty or too large (max 5 MB)")
    try:
        image = decode(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_predictor().predict(image)
