import os
import argparse

parser = argparse.ArgumentParser(description="Force YOLO label class IDs to 0 recursively under labels_root")
parser.add_argument("--labels_root", default="./labels", help="Path to labels root folder (default: ./labels)")
parser.add_argument("--dry_run", action="store_true", help="Don't write files, just report what would change")
args = parser.parse_args()

labels_root = os.path.abspath(args.labels_root)

fixed_files = 0
skipped_files = 0
modified_lines = 0
removed_lines = 0

if not os.path.isdir(labels_root):
    print(f"[error] labels_root not found: {labels_root}")
    raise SystemExit(1)

all_txt_files = []
for dirpath, _dirnames, filenames in os.walk(labels_root):
    for fn in filenames:
        if fn.endswith('.txt'):
            all_txt_files.append(os.path.join(dirpath, fn))

if not all_txt_files:
    print(f"[warn] No .txt files found under {labels_root}")

for file_path in sorted(all_txt_files):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"[warn] Failed to read {file_path}: {e}")
        skipped_files += 1
        continue

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            removed_lines += 1
            continue
        parts[0] = "0"
        new_lines.append(" ".join(parts))

    if args.dry_run:
        if new_lines != lines:
            fixed_files += 1
            modified_lines += len(new_lines)
        else:
            skipped_files += 1
        continue

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        if new_lines:
            fixed_files += 1
            modified_lines += len(new_lines)
        else:
            skipped_files += 1
    except Exception as e:
        print(f"[warn] Failed to write {file_path}: {e}")
        skipped_files += 1

print(f"Updated {fixed_files} label files, skipped {skipped_files}.")
print(f"Lines modified: {modified_lines}, lines removed (invalid): {removed_lines}.")
