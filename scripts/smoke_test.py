"""End-to-end check of the whole pipeline on tiny SYNTHETIC images (no download, no internet).
    python scripts/smoke_test.py
It trains a small model, predicts, uploads data through the API, retrains and evaluates."""
import io
import os
import sys
import tempfile
import time
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ.update(DATA_DIR=str(tmp / "data"), MODELS_DIR=str(tmp / "models"), PLANT_WEIGHTS="none",
                  AUTO_RETRAIN="false", RETRAIN_EPOCHS="1", BATCH_SIZE="8", TF_CPP_MIN_LOG_LEVEL="3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CLASSES = ["Tomato_healthy", "Tomato_Early_blight", "Potato_Late_blight"]
COLOURS = [(40, 160, 50), (150, 110, 40), (110, 80, 140)]
rng = np.random.default_rng(0)


def make_img(colour):
    a = np.clip(np.array(colour) + rng.normal(0, 25, (224, 224, 3)), 0, 255).astype("uint8")
    return Image.fromarray(a)


def fill(split, n):
    for c, col in zip(CLASSES, COLOURS):
        d = tmp / "data" / split / c
        d.mkdir(parents=True)
        for i in range(n):
            make_img(col).save(d / f"{i}.jpg")


fill("train", 24)
fill("test", 8)

from src import config  # noqa: E402
from src.model import build_model, evaluate_model, save_model, train_model  # noqa: E402
from src.preprocessing import build_datasets, compute_class_weights  # noqa: E402

train_ds, val_ds, test_ds, names = build_datasets()
model = build_model(len(names))
train_model(model, train_ds, val_ds, compute_class_weights(config.TRAIN_DIR, names), 3, 1, 10)
m = evaluate_model(model, test_ds, names)
print("TEST METRICS:", {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
save_model(model, names)
assert config.MODEL_PATH.exists() and config.CLASS_NAMES_PATH.exists()

from fastapi.testclient import TestClient  # noqa: E402
from app.api import app  # noqa: E402

client = TestClient(app)
buf = io.BytesIO(); make_img(COLOURS[1]).save(buf, "JPEG")
r = client.post("/predict", files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")})
print("PREDICT:", r.status_code, r.json()["label"], round(r.json()["confidence"], 3)); assert r.status_code == 200
print("HEALTH:", client.get("/health").json())

imgs = []
for i in range(6):
    b = io.BytesIO(); make_img(COLOURS[0]).save(b, "PNG"); imgs.append(("files", (f"n{i}.png", b.getvalue(), "image/png")))
imgs.append(("files", ("broken.jpg", b"not an image", "image/jpeg")))
r = client.post("/upload", files=imgs, data={"label": CLASSES[0]})
print("UPLOAD:", r.json()); assert r.json()["saved"] == 7
assert client.post("/upload", files=imgs[:1], data={"label": "NOPE"}).status_code == 400

import zipfile  # noqa: E402
zb = io.BytesIO()
with zipfile.ZipFile(zb, "w") as z:
    for i in range(3):
        b = io.BytesIO(); make_img(COLOURS[2]).save(b, "JPEG"); z.writestr(f"bulk/{CLASSES[2]}/z{i}.jpg", b.getvalue())
print("ZIP:", client.post("/upload_zip", files={"file": ("d.zip", zb.getvalue(), "application/zip")}).json())

r = client.post("/retrain"); print("RETRAIN:", r.status_code, r.json())
for _ in range(120):
    run = client.get("/retrain/status").json()["latest_run"]
    if run and run["status"] != "running":
        break
    time.sleep(2)
print("RUN:", run); assert run["status"] == "finished", run
print("SUMMARY:", client.get("/uploads/summary").json())
print("EVALUATE:", client.post("/evaluate?max_samples=24").json())
print("STATS:", client.get("/data/stats").json())
print("FEATURES rows:", len(client.get("/data/features").json()))
print("\nSMOKE TEST PASSED")
