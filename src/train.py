"""Initial training from the command line (the notebook does the same step by step).
    python -m src.train --head-epochs 8 --finetune-epochs 10
"""
import argparse
import json

from src import config
from src.model import build_model, evaluate_model, save_model, train_model
from src.preprocessing import build_datasets, compute_class_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    args = ap.parse_args()

    train_ds, val_ds, test_ds, names = build_datasets()
    cw = compute_class_weights(config.TRAIN_DIR, names)
    model = build_model(len(names))
    train_model(model, train_ds, val_ds, cw, args.head_epochs, args.finetune_epochs)
    metrics = evaluate_model(model, test_ds, names)
    save_model(model, names)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "report")}, indent=2))


if __name__ == "__main__":
    main()
