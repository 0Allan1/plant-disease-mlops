"""Data processing: datasets, class weights, image cleaning and simple image features."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src import config


# ---------------------------------------------------------------- tf.data pipelines
def build_datasets(train_dir=config.TRAIN_DIR, test_dir=config.TEST_DIR, class_names=None,
                   img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, val_split=0.2, seed=config.SEED):
    """Return train / validation / test tf.data datasets (labels are one-hot).
    Pixel scaling happens INSIDE the model, so raw 0-255 images are fed here."""
    import tensorflow as tf
    common = dict(labels="inferred", label_mode="categorical", class_names=class_names,
                  image_size=img_size, batch_size=batch_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="training", seed=seed, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="validation", seed=seed, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **common)
    names = train_ds.class_names
    auto = tf.data.AUTOTUNE
    return (train_ds.prefetch(auto), val_ds.prefetch(auto), test_ds.prefetch(auto), names)


def class_distribution(data_dir) -> pd.Series:
    data_dir = Path(data_dir)
    return pd.Series({d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTS)
                      for d in sorted(data_dir.iterdir()) if d.is_dir()}, name="images")


def compute_class_weights(train_dir, class_names):
    """Balanced class weights: rare classes (e.g. healthy potato) count more in the loss."""
    counts = class_distribution(train_dir).reindex(class_names).fillna(0)
    total, k = counts.sum(), len(class_names)
    return {i: float(total / (k * max(c, 1))) for i, c in enumerate(counts)}


# ---------------------------------------------------------------- single image helpers
def prepare_image(image, img_size=config.IMG_SIZE) -> np.ndarray:
    """Open an image (path, bytes or file-like), force RGB, resize. Returns (1,H,W,3) float32 0-255."""
    import io
    if isinstance(image, (bytes, bytearray)):
        image = io.BytesIO(image)
    with Image.open(image) as im:
        im = im.convert("RGB").resize(img_size)
        return np.expand_dims(np.asarray(im, dtype="float32"), 0)


def validate_and_standardise(src_path, dst_path, img_size=config.IMG_SIZE) -> tuple[bool, str]:
    """Used on uploaded training data: reject corrupt files, convert to RGB, resize, save as JPEG."""
    try:
        with Image.open(src_path) as im:
            im.verify()
        with Image.open(src_path) as im:
            im = im.convert("RGB").resize(img_size)
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst_path, "JPEG", quality=95)
        return True, "ok"
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"rejected: {e}"


# ---------------------------------------------------------------- image features for the visualisations
def compute_image_features(data_dir, per_class=80, seed=config.SEED, small=(64, 64)) -> pd.DataFrame:
    """Colour features for a random sample of images per class.
    - mean_green : average green channel (0-255)
    - brightness : average of R,G,B
    - non_green_ratio : share of pixels where green is NOT the dominant channel
                        (brown / yellow lesions, soil and shadows)"""
    rng = np.random.default_rng(seed)
    rows = []
    for cls_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTS]
        if not files:
            continue
        pick = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
        for i in pick:
            try:
                with Image.open(files[i]) as im:
                    a = np.asarray(im.convert("RGB").resize(small), dtype="float32")
            except (UnidentifiedImageError, OSError):
                continue
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            rows.append({"class": cls_dir.name, "mean_red": r.mean(), "mean_green": g.mean(),
                         "mean_blue": b.mean(), "brightness": a.mean(),
                         "non_green_ratio": float(((g <= r) | (g <= b)).mean())})
    return pd.DataFrame(rows)
