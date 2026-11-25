"""Data acquisition: download PlantVillage (15 classes, ~20k leaf images) and create
the data/train and data/test folders (stratified 80/20 split).

Usage:
    python -m src.data_acquisition                     # download from Kaggle via kagglehub
    python -m src.data_acquisition --source /path/to/PlantVillage   # use a folder you already have
    python -m src.data_acquisition --max-per-class 300 # small subset (fast experiments)
"""
import argparse
import os
import random
import shutil
from pathlib import Path

from src import config


def download_plantvillage() -> Path:
    """Download the dataset with kagglehub (public dataset, no API key normally needed)."""
    import kagglehub
    return Path(kagglehub.dataset_download("emmarex/plantdisease"))


def _images_in(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS)


def find_class_root(root: Path) -> Path:
    """Find the folder whose sub-folders are the classes (the download is nested differently
    depending on the version, so we search for it instead of hard-coding a path)."""
    best, best_n = None, 0
    for dirpath, dirnames, _ in os.walk(root):
        n = sum(1 for d in dirnames if _images_in(Path(dirpath) / d))
        if n > best_n:
            best, best_n = Path(dirpath), n
    if best is None or best_n < 2:
        raise RuntimeError(f"No class folders with images found under {root}")
    return best


def split_and_copy(src_root: Path, train_dir: Path = config.TRAIN_DIR, test_dir: Path = config.TEST_DIR,
                   test_size: float = 0.2, max_per_class: int | None = None, seed: int = config.SEED):
    rng = random.Random(seed)
    counts = {}
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        imgs = _images_in(cls_dir)
        if not imgs:
            continue
        rng.shuffle(imgs)
        if max_per_class:
            imgs = imgs[:max_per_class]
        n_test = max(1, int(len(imgs) * test_size))
        parts = {"test": (test_dir, imgs[:n_test]), "train": (train_dir, imgs[n_test:])}
        for name, (base, files) in parts.items():
            out = base / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)
        counts[cls_dir.name] = (len(imgs) - n_test, n_test)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None, help="folder with one sub-folder per class")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-create data/train and data/test")
    args = ap.parse_args()

    if any(config.TRAIN_DIR.glob("*/*")) and not args.force:
        print("data/train already has images. Use --force to rebuild.")
        return
    if args.force:
        for d in (config.TRAIN_DIR, config.TEST_DIR):
            shutil.rmtree(d, ignore_errors=True)

    src = args.source or download_plantvillage()
    root = find_class_root(Path(src))
    print(f"Using class folders in: {root}")
    counts = split_and_copy(root, max_per_class=args.max_per_class)
    for cls, (tr, te) in counts.items():
        print(f"{cls:45s} train={tr:5d} test={te:4d}")
    print(f"Done: {len(counts)} classes.")


if __name__ == "__main__":
    main()
