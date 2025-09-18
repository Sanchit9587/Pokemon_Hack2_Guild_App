#!/usr/bin/env python3
"""
Split a YOLO dataset into train/val/test folders.

- Expects an images directory and a labels directory where label files are YOLO .txt with matching basenames.
- Creates images/train, images/val, images/test, labels/train, labels/val, labels/test and moves (or copies) matching pairs.

Usage examples:
  python split_yolo_dataset.py --images_dir images --labels_dir labels --train_ratio 0.85 --seed 42
  python split_yolo_dataset.py --images_dir images --labels_dir labels --copy --update-data-yaml data.yml

Notes:
- By default, images without a corresponding label are skipped. Use --allow-missing-labels to include them anyway.
- If folders already contain files, use --force to proceed.
"""

from __future__ import annotations
import argparse
import random
import shutil
from pathlib import Path
import sys
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_label_pairs(images_dir: Path, labels_dir: Path, allow_missing: bool) -> List[Tuple[Path, Path | None]]:
    """Return (image_path, label_path_or_None) pairs.

    - Recursively finds images under images_dir.
    - For each image, first look for a label that mirrors the relative path, e.g.:
        images/train/foo/bar.png -> labels/train/foo/bar.txt
      If not found, fallback to labels_dir/<stem>.txt for backward compatibility.
    """
    images: List[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(images_dir.rglob(f"*{ext}"))
    images = [p for p in images if p.is_file()]

    pairs: List[Tuple[Path, Path | None]] = []
    for img in images:
        try:
            rel = img.relative_to(images_dir)
        except ValueError:
            rel = Path(img.name)
        mirrored = (labels_dir / rel).with_suffix(".txt")
        root_lbl = labels_dir / (img.stem + ".txt")
        if mirrored.exists():
            pairs.append((img, mirrored))
        elif root_lbl.exists():
            pairs.append((img, root_lbl))
        elif allow_missing:
            pairs.append((img, None))
        else:
            continue
    return pairs


essential_subdirs = ["train", "val", "test"]


def ensure_empty_or_force(paths: List[Path], force: bool) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        if not force and any(p.iterdir()):
            raise SystemExit(f"Refusing to proceed: '{p}' is not empty. Use --force to continue.")


def split_indices(n: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(round(train_ratio * n))
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:]
    return train_idx, val_idx


def move_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def update_data_yaml(data_yaml: Path, images_dir: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        print(f"[info] PyYAML not installed; skipping update of {data_yaml}.", file=sys.stderr)
        print("       Set these keys manually:", file=sys.stderr)
        print(f"       train: {images_dir / 'train'}", file=sys.stderr)
        print(f"       val:   {images_dir / 'val'}", file=sys.stderr)
        return

    try:
        if data_yaml.exists():
            data = yaml.safe_load(data_yaml.read_text()) or {}
        else:
            data = {}
        data["train"] = str(images_dir / "train")
        data["val"] = str(images_dir / "val")
        data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"[ok] Updated {data_yaml} with train/val paths.")
    except Exception as e:
        print(f"[warn] Failed to update {data_yaml}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test.")
    ap.add_argument("--images_dir", type=Path, default=Path("images"), help="Path to images directory (default: images)")
    ap.add_argument("--labels_dir", type=Path, default=Path("labels"), help="Path to labels directory (default: labels)")
    # Defaults updated to 0.7 (train), 0.2 (test), 0.1 (val)
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Proportion for training (default: 0.7)")
    ap.add_argument("--val_ratio", type=float, default=None, help="Proportion for validation (default: 0.1 if --test_ratio > 0 else remainder)")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Proportion for test (default: 0.2)")
    # Convenience: --ratios TRAIN TEST VAL
    ap.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN","TEST","VAL"), help="Three ratios in order: train test val; will override other ratio flags")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    ap.add_argument("--force", action="store_true", help="Proceed even if destination folders are not empty")
    ap.add_argument("--allow-missing-labels", action="store_true", help="Include images with no label file")
    ap.add_argument("--update-data-yaml", type=Path, default=None, help="Optional path to data.yml to update train/val/test")
    ap.add_argument("--debug", action="store_true", help="Print debug information about discovered files and matching")
    ap.add_argument("--dry_run", action="store_true", help="Plan split only; don't move/copy files or update data.yml")
    args = ap.parse_args()

    images_dir: Path = args.images_dir
    labels_dir: Path = args.labels_dir

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    # discover images for debug
    if args.debug:
        discovered = []
        for ext in IMAGE_EXTS:
            discovered.extend([p for p in images_dir.rglob(f"*{ext}") if p.is_file()])
        print(f"[debug] images found under {images_dir}: {len(discovered)}")
        if len(discovered) > 0:
            print("[debug] first 5 images:")
            for p in discovered[:5]:
                print(f"   - {p}")

    pairs = find_image_label_pairs(images_dir, labels_dir, allow_missing=args.allow_missing_labels)

    if args.debug:
        print(f"[debug] matched image/label pairs: {len(pairs)}")
        # show a few diagnostics about label resolution for first few discovered images
        try:
            sample = discovered[:5]
        except Exception:
            sample = []
        for img in sample:
            try:
                rel = img.relative_to(images_dir)
            except Exception:
                rel = Path(img.name)
            mirrored = (labels_dir / rel).with_suffix(".txt")
            root_lbl = labels_dir / (img.stem + ".txt")
            status = ""
            if mirrored.exists():
                status = f"OK (mirrored) -> {mirrored}"
            elif root_lbl.exists():
                status = f"OK (root) -> {root_lbl}"
            else:
                status = "MISSING"
            print(f"[debug] label for {img.name}: {status}")

    if not pairs:
        msg = "No images found to split. Check your directories and extensions."
        if args.debug:
            msg += " Also verify that matching .txt label files exist under the labels directory."
        raise SystemExit(msg)

    # determine ratios
    if args.ratios is not None:
        tr, te, va = args.ratios
    else:
        tr = float(args.train_ratio)
        te = float(args.test_ratio or 0.0)
        if args.val_ratio is None:
            # if test set requested, default val to 0.1, else fill remainder
            va = 0.1 if te > 0 else max(0.0, 1.0 - tr)
        else:
            va = float(args.val_ratio)
    total = tr + va + te
    if not (0.99 <= total <= 1.01):
        # normalize to sum to 1 to be forgiving
        tr, va, te = [x/total for x in (tr, va, te)]
    if any(x < 0 for x in (tr, va, te)):
        raise SystemExit("Ratios must be non-negative and sum to 1.")

    # Destination dirs
    img_train = images_dir / "train"
    img_val = images_dir / "val"
    img_test = images_dir / "test" if te > 0 else None

    lbl_train = labels_dir / "train"
    lbl_val = labels_dir / "val"
    lbl_test = labels_dir / "test" if te > 0 else None

    # split indices 3-way
    n = len(pairs)
    idxs = list(range(n))
    random.Random(args.seed).shuffle(idxs)
    n_train = int(round(tr * n))
    n_val = int(round(va * n))
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:] if te > 0 else []

    if args.dry_run:
        print(f"[dry-run] Would split {n} images as: Train={len(train_idx)}, Val={len(val_idx)}" + (f", Test={len(test_idx)}" if te > 0 else ""))
        print(f"[dry-run] Train ratio={tr:.3f}, Val ratio={va:.3f}, Test ratio={te:.3f}")
        return

    dests = [img_train, img_val, lbl_train, lbl_val]
    if te > 0:
        dests += [img_test, lbl_test]  # type: ignore[list-item]
    ensure_empty_or_force([p for p in dests if p is not None], force=args.force)  # type: ignore[arg-type]

    def place(idx_list: List[int], img_dst: Path, lbl_dst: Path) -> int:
        moved = 0
        for i in idx_list:
            img, lbl = pairs[i]
            img_target = img_dst / (img.relative_to(images_dir)) if img.is_relative_to(images_dir) else img_dst / img.name
            img_target = img_target.with_suffix(img.suffix)
            move_or_copy(img, img_target, copy=args.copy)
            if lbl is not None:
                # mirror relative structure for labels as well
                try:
                    rel_img = img.relative_to(images_dir)
                    lbl_rel = rel_img.with_suffix('.txt')
                    lbl_target = lbl_dst / lbl_rel
                except ValueError:
                    lbl_target = lbl_dst / lbl.name
                move_or_copy(lbl, lbl_target, copy=args.copy)
            moved += 1
        return moved

    n_train = place(train_idx, img_train, lbl_train)
    n_val = place(val_idx, img_val, lbl_val)
    n_test = 0
    if te > 0 and img_test is not None and lbl_test is not None:
        n_test = place(test_idx, img_test, lbl_test)

    print(f"Done. Train: {n_train} images, Val: {n_val} images" + (f", Test: {n_test} images" if te > 0 else ""))

    if args.update_data_yaml is not None:
        try:
            import yaml
            data = {}
            if args.update_data_yaml.exists():
                data = yaml.safe_load(args.update_data_yaml.read_text()) or {}
            data["train"] = str(img_train)
            data["val"] = str(img_val)
            if te > 0 and img_test is not None:
                data["test"] = str(img_test)
            args.update_data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
            print(f"[ok] Updated {args.update_data_yaml} with train/val" + ("/test" if te > 0 else "") + " paths.")
        except Exception as e:
            print(f"[warn] Failed to update {args.update_data_yaml}: {e}", file=sys.stderr)
    else:
        print("Tip: Update your data.yaml with:")
        print(f"train: {img_train}")
        print(f"val:   {img_val}")
        if te > 0 and img_test is not None:
            print(f"test:  {img_test}")


if __name__ == "__main__":
    main()
