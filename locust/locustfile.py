"""Flood the prediction endpoint.   locust -f locust/locustfile.py --host http://localhost:8000
Set LOCUST_IMAGE to a leaf photo, otherwise the first image in data/test is used."""
import os
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent


def find_image():
    p = os.getenv("LOCUST_IMAGE")
    if p:
        return Path(p)
    for f in sorted((ROOT / "data" / "test").rglob("*")):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return f
    raise SystemExit("No image found. Set LOCUST_IMAGE=/path/to/leaf.jpg")


IMAGE_BYTES = find_image().read_bytes()


class PlantUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(10)
    def predict(self):
        self.client.post("/predict", files={"file": ("leaf.jpg", IMAGE_BYTES, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
