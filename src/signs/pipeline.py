"""End-to-end steps: classical → cnn → robustness → report. Each step saves what later steps need."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import joblib
import keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from signs import classical, evaluate, robustness
from signs.cnn import build_cnn
from signs.config import Config
from signs.data import Dataset, load_dataset

log = logging.getLogger(__name__)


def _paths(cfg: Config) -> dict[str, Path]:
    art, rep = Path(cfg.artifacts_dir), Path(cfg.reports_dir)
    art.mkdir(parents=True, exist_ok=True)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    return {
        "classical": art / "classical.joblib",
        "cnn": art / "cnn.keras",
        "preds": art / "test_preds.npz",
        "metrics": rep / "metrics.json",
        "figures": rep / "figures",
    }


def _update(path: Path, key: str, value) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2))


def _save_preds(path: Path, **arrays) -> None:
    existing = dict(np.load(path)) if path.exists() else {}
    existing.update(arrays)
    np.savez(path, **existing)


def _softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _per_image_ms(predict, images: np.ndarray, n: int = 200) -> float:
    predict(images[:5])  # warm-up
    start = time.perf_counter()
    for img in images[:n]:
        predict(img[None])
    return round((time.perf_counter() - start) / n * 1000, 2)


def leakage_check(ds: Dataset, cfg: Config, kind: str, c: float) -> dict:
    """Same model, two validation protocols: random frames vs whole tracks. Which one predicts the test score?"""
    x = np.concatenate([ds.x_train, ds.x_val])
    y = np.concatenate([ds.y_train, ds.y_val])
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=len(ds.y_val), random_state=cfg.seed, stratify=y)
    random_val = accuracy_score(y_va, classical.build_pipeline(kind, c, cfg).fit(x_tr, y_tr).predict(x_va))
    return {"random_frame_split_val_accuracy": round(float(random_val), 4)}


def step_classical(cfg: Config, ds: Dataset) -> None:
    p = _paths(cfg)
    results, kind, c = classical.ablation(ds, cfg)
    start = time.perf_counter()
    model = classical.build_pipeline(kind, c, cfg).fit(ds.x_train, ds.y_train)
    train_seconds = time.perf_counter() - start
    joblib.dump(model, p["classical"])
    scores = model.decision_function(ds.x_test)
    pred = scores.argmax(1)
    _save_preds(p["preds"], classical=pred, classical_conf=_softmax(scores).max(1), y=ds.y_test)
    info = {
        "ablation_val_accuracy": results,
        "best": {"features": kind, "C": c},
        "n_features": int(model.named_steps["features"].transform(ds.x_train[:1]).shape[1]),
        "train_seconds": round(train_seconds, 1),
        "val_accuracy_track_split": results[kind][c],
        **leakage_check(ds, cfg, kind, c),
        "test": evaluate.metrics(ds.y_test, pred),
        "ms_per_image": _per_image_ms(model.predict, ds.x_test),
        "size_mb": round(p["classical"].stat().st_size / 1e6, 2),
    }
    _update(p["metrics"], "classical", info)
    log.info("classical %s", json.dumps({k: v for k, v in info.items() if k != "ablation_val_accuracy"}))


def step_cnn(cfg: Config, ds: Dataset) -> None:
    p = _paths(cfg)
    keras.utils.set_random_seed(cfg.seed)
    model = build_cnn(cfg.cnn, cfg.data.image_size)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg.cnn.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-5),
    ]
    start = time.perf_counter()
    hist = model.fit(
        ds.x_train, ds.y_train, validation_data=(ds.x_val, ds.y_val), epochs=cfg.cnn.epochs, batch_size=cfg.cnn.batch_size, callbacks=callbacks, verbose=2
    )
    train_seconds = time.perf_counter() - start
    model.save(p["cnn"])
    probs = model.predict(ds.x_test, batch_size=512, verbose=0)
    pred = probs.argmax(1)
    _save_preds(p["preds"], cnn=pred, cnn_conf=probs.max(1))
    history = {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()}
    info = {
        "params": int(model.count_params()),
        "epochs_run": len(history["loss"]),
        "best_val_accuracy": max(history["val_accuracy"]),
        "train_seconds": round(train_seconds, 1),
        "test": evaluate.metrics(ds.y_test, pred),
        "ms_per_image": _per_image_ms(lambda b: model(b, training=False), ds.x_test),
        "size_mb": round(p["cnn"].stat().st_size / 1e6, 2),
        "history": history,
    }
    _update(p["metrics"], "cnn", info)
    log.info("cnn %s", json.dumps({k: v for k, v in info.items() if k != "history"}))


def step_robustness(cfg: Config, ds: Dataset) -> None:
    p = _paths(cfg)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(ds.y_test), 3000, replace=False)
    x, y = ds.x_test[idx], ds.y_test[idx]
    clf = joblib.load(p["classical"])
    cnn = keras.models.load_model(p["cnn"])

    def accs(images: np.ndarray) -> dict:
        return {
            "classical": round(float((clf.predict(images) == y).mean()), 4),
            "cnn": round(float((cnn.predict(images, batch_size=512, verbose=0).argmax(1) == y).mean()), 4),
        }

    clean = accs(x)
    rows = []
    for name in robustness.CORRUPTIONS:
        for s in cfg.robustness.severities:
            rows.append({"corruption": name, "severity": s, **accs(robustness.corrupt(x, name, s, seed=cfg.seed))})
            log.info("%s", rows[-1])
    mean_drop = {m: round(clean[m] - float(np.mean([r[m] for r in rows])), 4) for m in ("classical", "cnn")}
    _update(p["metrics"], "robustness", {"n_images": len(y), "clean": clean, "rows": rows, "mean_accuracy_drop": mean_drop})
    evaluate.plot_robustness(rows, clean, p["figures"] / "robustness.png")
    evaluate.plot_corruption_examples(x[1], robustness.corrupt, list(robustness.CORRUPTIONS), p["figures"] / "corruptions.png")


def step_report(cfg: Config, ds: Dataset) -> None:
    p = _paths(cfg)
    d = np.load(p["preds"])
    y, pc, pn = d["y"], d["classical"], d["cnn"]
    metrics = json.loads(p["metrics"].read_text())
    analysis = {
        "accuracy_ci95": {"classical": evaluate.bootstrap_ci(y, pc, seed=cfg.seed), "cnn": evaluate.bootstrap_ci(y, pn, seed=cfg.seed)},
        "mcnemar_classical_vs_cnn": evaluate.mcnemar(y, pc, pn),
        "worst_classes": {"classical": evaluate.worst_classes(y, pc), "cnn": evaluate.worst_classes(y, pn)},
        "confusion_pairs": {"classical": evaluate.confusion_pairs(y, pc), "cnn": evaluate.confusion_pairs(y, pn)},
        "accuracy_by_sign_size": evaluate.accuracy_by_size(y, {"classical": pc, "cnn": pn}, ds.side_test),
        "both_wrong": int(((pc != y) & (pn != y)).sum()),
        "errors": {"classical": int((pc != y).sum()), "cnn": int((pn != y).sum())},
    }
    _update(p["metrics"], "analysis", analysis)
    f = p["figures"]
    evaluate.plot_examples(ds.x_train, ds.y_train, f / "examples.png")
    darkest = ds.x_train[np.argsort(ds.x_train.reshape(len(ds.x_train), -1).mean(1))[len(ds.x_train) // 50]]
    evaluate.plot_preprocessing(darkest, classical.equalize(darkest, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))), f / "preprocessing.png")
    evaluate.plot_errors(ds.x_test, y, pc, d["classical_conf"], "HOG + SVM: most confident errors", f / "errors_classical.png")
    evaluate.plot_errors(ds.x_test, y, pn, d["cnn_conf"], "CNN: most confident errors", f / "errors_cnn.png")
    evaluate.plot_history(metrics["cnn"]["history"], f / "cnn_history.png")
    log.info("analysis %s", json.dumps(analysis, indent=1))


STEPS = {"classical": step_classical, "cnn": step_cnn, "robustness": step_robustness, "report": step_report}


def run(cfg: Config, steps: list[str]) -> None:
    ds = load_dataset(cfg)
    for name in steps:
        start = time.perf_counter()
        STEPS[name](cfg, ds)
        log.info("step %s finished in %.0f s", name, time.perf_counter() - start)
