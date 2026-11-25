"""Prediction: loads the saved model once and reloads it automatically when the file changes
(so every API container picks up a model that was just retrained)."""
import threading
import time

import numpy as np

from src import config
from src.preprocessing import prepare_image


class Predictor:
    def __init__(self):
        self._model = None
        self._classes = None
        self._mtime = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Train one first (notebook or python -m src.train).")
        mtime = config.MODEL_PATH.stat().st_mtime
        if self._model is None or mtime != self._mtime:
            with self._lock:
                if self._model is None or mtime != self._mtime:
                    import json
                    import keras
                    self._model = keras.models.load_model(config.MODEL_PATH, compile=False)
                    self._classes = json.loads(config.CLASS_NAMES_PATH.read_text())
                    self._mtime = mtime

    @property
    def loaded(self):
        return self._model is not None

    @property
    def version(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._mtime)) if self._mtime else None

    @property
    def class_names(self):
        self._ensure_loaded()
        return list(self._classes)

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def predict(self, image, top_k=3):
        """image: bytes, file-like or path. Returns predicted label, confidence and the top-k classes."""
        self._ensure_loaded()
        probs = self._model(prepare_image(image), training=False).numpy()[0]
        order = np.argsort(probs)[::-1]
        top = [{"label": self._classes[i], "confidence": float(probs[i])} for i in order[:top_k]]
        return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top": top}


_predictor = Predictor()


def get_predictor() -> Predictor:
    return _predictor


def predict_image(image, top_k=3):
    return _predictor.predict(image, top_k)
