# inference.py — using RF‑DETR for detection

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from rfdetr import RFDETRBase, RFDETRSmall  # RF‑DETR library

# Path to your trained RF‑DETR checkpoint
best_ckpt = "./runs/train_20251108_0054402/weights/best.pt"

# Choose model variant
model = RFDETRSmall.load(best_ckpt)  # or RFDETRBase.load(best_ckpt)

# Directory containing images for inference
source = "./data/images/test"

# Function to convert RF‑DETR prediction output to COCO‑style
def _to_coco_format(dets, image_id):
    """
    dets: list of dicts from model.predict(...)
    returns list of dicts: {image_id, category_id, bbox, score}
    """
    out = []
    for d in dets:
        category_id = int(d["category_id"])
        bbox = d["bbox"]  # should be [x, y, w, h]
        score = float(d["score"])
        out.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": score
        })
    return out

# Inference pipeline
results_out = []
for img_path in tqdm(sorted(Path(source).glob("*.*"))):
    image_id = None
    stem = img_path.stem
    try:
        image_id = int(stem)
    except:
        image_id = stem

    img = Image.open(img_path).convert("RGB")
    dets = model.predict(img, threshold=0.5)  # adjust threshold as needed

    if not dets:
        continue

    coco_preds = _to_coco_format(dets, image_id)
    results_out.extend(coco_preds)

# Save to JSON
timestamp = time.strftime('%Y%m%d_%H%M%S')
out_file = f"inference_results_rfdetr_{timestamp}.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results_out, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results_out)} detections to {out_file}")
