"""FastAPI service: prediction, data upload, retraining, health/up-time, evaluation and data statistics.
Run locally:  uvicorn app.api:app --reload      (interactive docs at /docs)
"""
import io
import logging
import os
import random
import socket
import time
import uuid
import zipfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src import config, db  # noqa: E402
from src.prediction import get_predictor  # noqa: E402
from src.preprocessing import class_distribution, compute_image_features, prepare_image  # noqa: E402
from src.retrain import start_retraining_in_background  # noqa: E402

app = FastAPI(title="Plant Disease Classifier API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
db.init_db()

STARTED = time.time()
STATS = {"requests": 0, "predictions": 0, "latency_sum": 0.0, "errors": 0}
MAX_ZIP_FILES, MAX_ZIP_BYTES = 5000, 500 * 1024 * 1024
predictor = get_predictor()


@app.on_event("startup")
def warm_up():
    """Load the model and run one dummy prediction so the first real request is fast."""
    try:
        predictor.model(np.zeros((1, *config.IMG_SIZE, 3), dtype="float32"), training=False)
    except FileNotFoundError:
        logging.getLogger("api").warning("No model yet - train one first.")


@app.middleware("http")
async def count_requests(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        STATS["errors"] += 1
        raise
    STATS["requests"] += 1
    STATS["latency_sum"] += time.perf_counter() - t0
    if response.status_code >= 500:
        STATS["errors"] += 1
    return response


def _known_classes():
    if config.CLASS_NAMES_PATH.exists():
        return predictor.class_names
    return sorted(d.name for d in config.TRAIN_DIR.iterdir() if d.is_dir()) if config.TRAIN_DIR.exists() else []


# ------------------------------------------------------------------ monitoring
@app.get("/health")
def health():
    n = max(STATS["requests"], 1)
    return {"status": "ok", "container": socket.gethostname(), "uptime_seconds": round(time.time() - STARTED, 1),
            "model_file_exists": config.MODEL_PATH.exists(), "model_loaded": predictor.loaded,
            "model_version": predictor.version, "requests_served": STATS["requests"],
            "predictions_served": STATS["predictions"], "errors": STATS["errors"],
            "avg_latency_ms": round(1000 * STATS["latency_sum"] / n, 2)}


@app.get("/classes")
def classes():
    return {"classes": _known_classes()}


# ------------------------------------------------------------------ prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        result = predictor.predict(data)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # unreadable image etc.
        raise HTTPException(400, f"Could not process image: {e}")
    STATS["predictions"] += 1
    return {"filename": file.filename, "container": socket.gethostname(), **result}


# ------------------------------------------------------------------ data upload (for retraining)
def _save_upload(data: bytes, label: str, original_name: str) -> int:
    folder = config.UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{Path(original_name).suffix.lower() or '.jpg'}"
    path.write_bytes(data)
    return db.add_upload(original_name, label, path)


def _maybe_auto_retrain():
    pending = len(db.pending_uploads())
    triggered = False
    if config.AUTO_RETRAIN and pending >= config.RETRAIN_THRESHOLD:
        triggered = start_retraining_in_background("auto-threshold")
    return pending, triggered


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), label: str = Form(...)):
    """Bulk upload of several images that all belong to one class."""
    if label not in _known_classes():
        raise HTTPException(400, f"Unknown label '{label}'. Valid labels: {_known_classes()}")
    saved = 0
    for f in files:
        if Path(f.filename).suffix.lower() in config.IMAGE_EXTS:
            _save_upload(await f.read(), label, f.filename)
            saved += 1
    pending, triggered = _maybe_auto_retrain()
    return {"saved": saved, "skipped": len(files) - saved, "pending_total": pending, "auto_retrain_started": triggered,
            "retrain_threshold": config.RETRAIN_THRESHOLD}


@app.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    """Upload a .zip that contains one folder per class: leaf_disease/<class_name>/image.jpg"""
    classes_ = set(_known_classes())
    raw = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Not a valid zip file")
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_ZIP_FILES or sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
        raise HTTPException(413, "Zip too large")
    saved, skipped, skipped_examples = 0, 0, []
    for info in infos:
        name = info.filename.replace("\\", "/")   # zips made by Windows PowerShell use backslashes
        parts = Path(name).parts
        label = parts[-2] if len(parts) >= 2 else None
        if label in classes_ and Path(name).suffix.lower() in config.IMAGE_EXTS:
            _save_upload(zf.read(info), label, Path(name).name)
            saved += 1
        else:
            skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(name)
    pending, triggered = _maybe_auto_retrain()
    return {"saved": saved, "skipped": skipped, "skipped_examples": skipped_examples,
            "pending_total": pending, "auto_retrain_started": triggered,
            "retrain_threshold": config.RETRAIN_THRESHOLD}


@app.get("/uploads/summary")
def uploads_summary():
    return {**db.upload_counts(), "retrain_threshold": config.RETRAIN_THRESHOLD}


# ------------------------------------------------------------------ retraining
@app.post("/retrain")
def retrain():
    if not config.MODEL_PATH.exists():
        raise HTTPException(409, "No base model yet. Train the first model before retraining.")
    if not start_retraining_in_background("manual"):
        raise HTTPException(409, "A retraining run is already in progress.")
    return {"message": "Retraining started in the background. Check /retrain/status."}


@app.get("/retrain/status")
def retrain_status():
    return {"latest_run": db.latest_run(), "history": db.list_runs(5), "pending_uploads": len(db.pending_uploads())}


# ------------------------------------------------------------------ evaluation in production + data insights
@app.post("/evaluate")
def evaluate(max_samples: int = 300):
    """Evaluate the DEPLOYED model on a random sample of the held-out test set."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    names = predictor.class_names
    files = [(f, i) for i, c in enumerate(names) if (config.TEST_DIR / c).exists()
             for f in (config.TEST_DIR / c).iterdir() if f.suffix.lower() in config.IMAGE_EXTS]
    if not files:
        raise HTTPException(409, "No test images found in data/test")
    random.Random(config.SEED).shuffle(files)
    files = files[:max_samples]
    t0 = time.perf_counter()
    y_true, y_pred = [], []
    model = predictor.model
    for start in range(0, len(files), 32):
        batch = files[start:start + 32]
        x = np.concatenate([prepare_image(f) for f, _ in batch])
        y_pred += model(x, training=False).numpy().argmax(1).tolist()
        y_true += [i for _, i in batch]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"samples": len(y_true), "accuracy": accuracy_score(y_true, y_pred), "precision_macro": p,
            "recall_macro": r, "f1_macro": f1, "seconds": round(time.perf_counter() - t0, 2),
            "ms_per_image": round(1000 * (time.perf_counter() - t0) / len(y_true), 2),
            "model_version": predictor.version}


@app.get("/data/stats")
def data_stats():
    out = {"train": {}, "test": {}}
    for split, path in (("train", config.TRAIN_DIR), ("test", config.TEST_DIR)):
        if path.exists() and any(path.iterdir()):
            out[split] = class_distribution(path).to_dict()
    return out


@app.get("/data/features")
def data_features(refresh: bool = False):
    """Colour features per class (cached) used for the visualisations."""
    import pandas as pd
    if refresh or not config.FEATURE_CACHE.exists():
        if not config.TRAIN_DIR.exists() or not any(config.TRAIN_DIR.iterdir()):
            raise HTTPException(409, "data/train is empty")
        compute_image_features(config.TRAIN_DIR).to_csv(config.FEATURE_CACHE, index=False)
    return pd.read_csv(config.FEATURE_CACHE).to_dict(orient="records")
