"""Model creation, training, evaluation and saving.

Architecture: MobileNetV2 (ImageNet pre-trained) + small classification head.
Optimisation techniques used: transfer learning, data augmentation, dropout, L2 regularisation,
class weights, Adam with a lower fine-tuning learning rate, EarlyStopping, ReduceLROnPlateau.
"""
import json
import shutil
import time

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config


def build_model(num_classes: int, img_size=config.IMG_SIZE, dropout=0.3, l2=1e-4, weights=config.BASE_WEIGHTS):
    import keras
    from keras import layers
    inputs = keras.Input(shape=(*img_size, 3), name="image")
    # augmentation layers are only active while training
    x = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.1),
                          layers.RandomZoom(0.1), layers.RandomContrast(0.1)], name="augmentation")(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(x)      # MobileNetV2 expects [-1, 1]
    base = keras.applications.MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights=weights)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")


def get_base_model(model):
    import keras
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name != "augmentation":
            return layer
    raise ValueError("Base model not found")


def compile_model(model, lr=1e-3):
    import keras
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model


def unfreeze_top(model, n_layers=40):
    """Unfreeze the last n layers of the base model. BatchNorm layers stay frozen (more stable)."""
    import keras
    base = get_base_model(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def default_callbacks(patience=3):
    import keras
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7)]


def train_model(model, train_ds, val_ds, class_weight=None, head_epochs=8, finetune_epochs=10, unfreeze_layers=40):
    """Phase 1: train the new head with the base frozen. Phase 2: fine-tune the top of the base at a low LR."""
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=class_weight, callbacks=default_callbacks())
    unfreeze_top(model, unfreeze_layers)
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                   class_weight=class_weight, callbacks=default_callbacks())
    return h1.history, h2.history


def predict_probs(model, ds):
    """Return (y_true, probs) for a (non-shuffled) one-hot dataset."""
    y_true, probs = [], []
    for x, y in ds:
        probs.append(model(x, training=False).numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true).argmax(1), np.concatenate(probs)


def evaluate_model(model, ds, class_names):
    """Loss, accuracy, precision, recall, F1 (macro + weighted), per-class report, confusion matrix."""
    loss = float(model.evaluate(ds, verbose=0, return_dict=True)["loss"])
    y_true, probs = predict_probs(model, ds)
    y_pred = probs.argmax(1)
    labels = list(range(len(class_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    return {
        "loss": loss, "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        zero_division=0, output_dict=True),
        "n_samples": int(len(y_true)),
    }


def save_model(model, class_names):
    """Save the model + class names. The previous model is archived, never overwritten blindly."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if config.MODEL_PATH.exists():
        config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.MODEL_PATH, config.ARCHIVE_DIR / f"model_{time.strftime('%Y%m%d_%H%M%S')}.h5")
    # Re-building a clean copy before saving avoids a Keras 3 bug where a model that was itself
    # loaded from .h5 cannot be written back to .h5. The weights are identical.
    clean = build_model(len(class_names), weights=None)
    clean.set_weights(model.get_weights())
    clean.save(config.MODEL_PATH)
    config.CLASS_NAMES_PATH.write_text(json.dumps(list(class_names)))


def load_class_names():
    return json.loads(config.CLASS_NAMES_PATH.read_text())
