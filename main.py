# main.py — RF-DETR train + val predict + COCO eval (no YOLO)

import os
import json
import glob
from PIL import Image

from rfdetr import RFDETRBase, RFDETRSmall
from train import train_rfdetr_model  # RF-DETR trainer with COCO oversampling

# ---------------- Config ----------------
DATASET_DIR    = "./data/images"  # must contain train/, val/, annotations/instances_*.json
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "val")
VAL_ANN        = os.path.join(DATASET_DIR, "annotations", "instances_val.json")
PRED_JSON      = "./predictions_rfdetr_val.json"

# Optional: set a checkpoint path to resume
best_ckpt = ""  # e.g., "./runs/rfdetr_20251110_130530/best.pth" (whatever RF-DETR saves)


def _load_coco_image_id_map(val_json_path):
    """Map basename(file_name) -> image_id from COCO 'images' list."""
    with open(val_json_path, "r") as f:
        js = json.load(f)
    by_name = {}
    for im in js.get("images", []):
        name = os.path.basename(im["file_name"])
        by_name[name] = im["id"]
    return by_name


def _rf_to_coco_preds(dets, image_id):
    """
    RF-DETR .predict() item(s) -> COCO dets:
    {image_id, category_id, bbox[x,y,w,h], score}
    """
    out = []
    for d in dets:
        cat = int(d.get("category_id", d.get("class_id", 0)))
        bbox = d.get("bbox", d.get("box", [0, 0, 0, 0]))
        score = float(d.get("score", d.get("confidence", 0.0)))
        out.append({"image_id": image_id, "category_id": cat, "bbox": [float(x) for x in bbox], "score": score})
    return out


def evaluate_with_coco(model):
    """Run RF-DETR predictions on val/ then call your eval.py."""
    print("[INFO] Generating validation predictions...")
    name2id = _load_coco_image_id_map(VAL_ANN)
    all_preds = []

    img_paths = sorted(glob.glob(os.path.join(VAL_IMAGES_DIR, "*.*")))
    for p in img_paths:
        im = Image.open(p).convert("RGB")
        # low threshold; COCO eval sweeps PR curve across thresholds
        dets = model.predict(im, threshold=0.001)
        image_id = name2id.get(os.path.basename(p))
        if image_id is None:
            stem = os.path.splitext(os.path.basename(p))[0]
            image_id = int(stem) if stem.isdigit() else None
        if image_id is None:
            print(f"[WARN] no image_id for {p}; skipping")
            continue
        all_preds.extend(_rf_to_coco_preds(dets, image_id))

    with open(PRED_JSON, "w") as f:
        json.dump(all_preds, f)
    print(f"[INFO] wrote predictions: {PRED_JSON} (n={len(all_preds)})")

    # Call your evaluator
    cmd = f"python eval.py --gt {VAL_ANN} --pred {PRED_JSON} --per-class"
    print(f"[INFO] Running COCO eval: {cmd}")
    os.system(cmd)


def main(is_freeze: bool = False):
    device = "cuda"
    print(device)

    # 1) Train / Resume
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"[INFO] Found checkpoint: {best_ckpt}")
        # Load + resume a bit to refine
        model = RFDETRSmall.load(best_ckpt)
        print("[INFO] Resuming fine-tuning for a few more epochs...")
        model.train(
            dataset_dir=DATASET_DIR,
            epochs=30 if not is_freeze else 1,
            batch_size=16,
            lr=1e-4,
            device=device,
        )
    else:
        print("[INFO] No previous checkpoint found. Starting new training...")
        # This uses oversampling internally (duplicates in COCO JSON during training)
        model, history, run_name = train_rfdetr_model(
            dataset_dir=DATASET_DIR,
            epochs=100,
            batch_size=16,
            lr=1e-4,
            device=device,
            cap=5,                 # oversample cap per image
            replace_instances=True,
            model_size="small",    # "small" or "base"
        )

    # 2) Evaluate on val with your COCO script
    #    (predict → save COCO dets → python eval.py ...)
    # If you have a dedicated "best" path RF-DETR produces, you could reload it here.
    if not (best_ckpt and os.path.exists(best_ckpt)):
        # use the in-memory `model` we just trained
        evaluate_with_coco(model)
    else:
        # re-open the provided checkpoint to ensure we eval exactly that
        model_eval = RFDETRBase.load(best_ckpt)
        evaluate_with_coco(model_eval)


if __name__ == "__main__":
    main(is_freeze=False)
