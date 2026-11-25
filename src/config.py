"""Central configuration. Every path can be overridden with an environment variable
so the same code runs locally, in Docker and in the cloud."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"          # raw files uploaded through the UI/API
FEATURE_CACHE = DATA_DIR / "feature_stats.csv"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "mlops.db"))

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "plant_disease_model.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
ARCHIVE_DIR = MODELS_DIR / "archive"       # previous model versions

IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Retraining trigger: when this many NEW images are waiting, retraining starts by itself.
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))

# "none" lets the smoke test run without downloading ImageNet weights.
BASE_WEIGHTS = None if os.getenv("PLANT_WEIGHTS", "imagenet").lower() == "none" else "imagenet"
