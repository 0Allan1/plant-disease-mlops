"""Retraining pipeline (triggered by the button / API or automatically by the upload threshold).

Steps: 1) take pending uploads from the database  2) preprocess them into data/train
       3) load the CURRENT model as the pre-trained starting point  4) fine-tune
       5) compare with the old model on the test set and only replace it if it is not worse.
"""
import logging
import shutil
import threading

from src import config, db
from src.model import compile_model, default_callbacks, evaluate_model, load_class_names, save_model, unfreeze_top
from src.preprocessing import build_datasets, compute_class_weights, validate_and_standardise

log = logging.getLogger("retrain")
ACCEPT_TOLERANCE = 0.005  # new model may be at most 0.5 points worse in accuracy and still be accepted


def integrate_uploads() -> tuple[int, int]:
    """Validate + standardise pending uploads and move them into data/train/<label>/."""
    ok = bad = 0
    for row in db.pending_uploads():
        dst = config.TRAIN_DIR / row["label"] / f"upload_{row['id']}.jpg"
        success, note = validate_and_standardise(row["path"], dst)
        db.mark_upload(row["id"], "processed" if success else "rejected", note)
        ok, bad = ok + success, bad + (not success)
    return ok, bad


def run_retraining(trigger="manual", epochs=config.RETRAIN_EPOCHS, lr=1e-5, unfreeze_layers=40):
    import keras
    run_id = db.start_run(trigger)
    if run_id is None:
        return {"started": False, "message": "A retraining run is already in progress."}
    try:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("No base model to retrain. Train the first model before retraining.")
        n_ok, n_bad = integrate_uploads()
        names = load_class_names()
        train_ds, val_ds, test_ds, names = build_datasets(class_names=names)
        cw = compute_class_weights(config.TRAIN_DIR, names)

        old = keras.models.load_model(config.MODEL_PATH, compile=False)
        compile_model(old, lr)
        old_m = evaluate_model(old, test_ds, names)

        # the current model IS the pre-trained model: copy its weights and keep training
        new = keras.models.load_model(config.MODEL_PATH, compile=False)
        unfreeze_top(new, unfreeze_layers)
        compile_model(new, lr)
        new.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, callbacks=default_callbacks(2))
        new_m = evaluate_model(new, test_ds, names)

        replaced = new_m["accuracy"] >= old_m["accuracy"] - ACCEPT_TOLERANCE
        if replaced:
            save_model(new, names)
        msg = (f"{n_ok} new images used, {n_bad} rejected. Accuracy {old_m['accuracy']:.4f} -> "
               f"{new_m['accuracy']:.4f}. New model {'ACCEPTED' if replaced else 'REJECTED (kept old model)'}.")
        db.finish_run(run_id, "finished", n_new_images=n_ok, old_accuracy=old_m["accuracy"],
                      new_accuracy=new_m["accuracy"], old_f1=old_m["f1_macro"], new_f1=new_m["f1_macro"],
                      model_replaced=int(replaced), message=msg)
        return {"started": True, "run_id": run_id, "message": msg}
    except Exception as e:  # noqa: BLE001 - we want every failure recorded in the DB
        log.exception("retraining failed")
        db.finish_run(run_id, "failed", message=str(e))
        return {"started": True, "run_id": run_id, "message": f"failed: {e}"}


def start_retraining_in_background(trigger="manual"):
    """Returns immediately; the API stays responsive while the model trains."""
    if db.latest_run() and db.latest_run()["status"] == "running":
        return False
    threading.Thread(target=run_retraining, kwargs={"trigger": trigger}, daemon=True).start()
    return True


if __name__ == "__main__":
    print(run_retraining("cli"))
