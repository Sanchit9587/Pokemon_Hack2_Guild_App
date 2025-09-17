import json
import os

# === Paths ===
json_file = "./annotations/instances_train.json"   # your JSON file
output_dir = "labels"                # where YOLO label txt files will go

os.makedirs(output_dir, exist_ok=True)

# === Load JSON ===
with open(json_file, "r") as f:
    data = json.load(f)

# === Build category mapping (COCO ids → YOLO ids) ===
cat2id = {cat["id"]: idx for idx, cat in enumerate(data["categories"])}

# === Group annotations by image_id ===
annotations_by_image = {}
for ann in data["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# === Process each image ===
for img in data["images"]:
    image_id = img["id"]
    file_name = os.path.splitext(img["file_name"])[0]  # no extension
    W, H = img["width"], img["height"]

    label_lines = []

    # If this image has annotations
    for ann in annotations_by_image.get(image_id, []):
        cat_id = cat2id[ann["category_id"]]  # re-map category
        x, y, w, h = ann["bbox"]

        # Convert COCO bbox → YOLO bbox
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        w_norm = w / W
        h_norm = h / H

        label_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Write label file (even if empty)
    out_path = os.path.join(output_dir, f"{file_name}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(label_lines))
