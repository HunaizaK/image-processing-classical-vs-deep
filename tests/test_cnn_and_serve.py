import cv2
import joblib
import keras
import numpy as np
import pytest
from fastapi.testclient import TestClient

from signs import NUM_CLASSES, serve
from signs.classical import build_pipeline
from signs.cnn import augmentation, build_cnn
from signs.config import load_config
from signs.evaluate import accuracy_by_size, mcnemar, worst_classes


def test_cnn_shapes_and_probabilities(toy_images):
    x, _ = toy_images
    model = build_cnn(load_config().cnn)
    probs = model.predict(x[:4], verbose=0)
    assert probs.shape == (4, NUM_CLASSES)
    np.testing.assert_allclose(probs.sum(1), 1, rtol=1e-5)


def test_cnn_learns_toy_signs(toy_images):
    keras.utils.set_random_seed(0)
    x, y = toy_images
    model = build_cnn(load_config().cnn)
    model.fit(x, y, epochs=15, batch_size=16, verbose=0)
    assert model.evaluate(x, y, verbose=0)[1] > 0.9


def test_augmentation_is_inactive_at_inference(toy_images):
    x, _ = toy_images
    aug = augmentation()
    batch = x[:4].astype("float32") / 255
    np.testing.assert_allclose(np.asarray(aug(batch, training=False)), batch, atol=1e-6)
    assert not any("Flip" in type(layer).__name__ for layer in aug.layers)  # flips change a sign's meaning


def test_mcnemar_and_size_buckets():
    y = np.array([0, 0, 1, 1, 2, 2])
    a = np.array([0, 0, 1, 0, 2, 0])
    b = np.array([0, 1, 1, 1, 2, 2])
    assert mcnemar(y, a, b)["only_a_correct"] == 1 and mcnemar(y, a, b)["only_b_correct"] == 2
    rows = accuracy_by_size(y, {"a": a}, np.array([10, 15, 25, 35, 50, 60]))
    assert [r["n"] for r in rows] == [2, 1, 1, 2]
    assert worst_classes(y, a, k=1)[0]["accuracy"] == 0.5


@pytest.fixture()
def client(toy_images, tmp_path, monkeypatch):
    x, y = toy_images
    joblib.dump(build_pipeline("hog_clahe_color", 0.1, load_config()).fit(x, y), tmp_path / "classical.joblib")
    monkeypatch.setattr(serve, "ARTIFACTS", tmp_path)
    monkeypatch.setenv("SIGNS_BACKEND", "classical")
    serve.get_predictor.cache_clear()
    yield TestClient(serve.app)
    serve.get_predictor.cache_clear()


def test_predict_endpoint_accepts_png_of_any_size(client, toy_images):
    x, y = toy_images
    # 3x nearest-neighbour upscale: the service's INTER_AREA downscale must recover the original pixels.
    big = cv2.resize(x[y == 1][0], (96, 96), interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    res = client.post("/predict", files={"file": ("sign.png", png.tobytes(), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["class_id"] == 1 and len(body["top3"]) == 3


def test_predict_rejects_non_images(client):
    assert client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 422
    assert client.post("/predict", files={"file": ("x.png", b"", "image/png")}).status_code == 422
