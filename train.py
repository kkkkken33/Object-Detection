# train.py — RF-DETR training with COCO oversampling (no YOLO deps)

import os
import json
import shutil
from datetime import datetime
from collections import Counter, defaultdict

from rfdetr import RFDETRBase, RFDETRSmall  # pip install rfdetr


# ---------------- COCO helpers ----------------

def _read_coco(ann_path: str):
    # os.makedirs(ann_path, exist_ok=True)
    with open(ann_path, "r") as f:
        coco = json.load(f)
    coco.setdefault("images", [])
    coco.setdefault("annotations", [])
    coco.setdefault("categories", [])
    return coco


def _write_coco(obj, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


def _build_imgid_to_classes(coco):
    """image_id -> set(class_ids) based on bbox segms (here: annotations list)."""
    img_to_cls = defaultdict(set)
    for ann in coco["annotations"]:
        img_to_cls[ann["image_id"]].add(int(ann["category_id"]))
    return img_to_cls


def _class_counts(coco):
    cnt = Counter()
    for ann in coco["annotations"]:
        cnt[int(ann["category_id"])] += 1
    return cnt


def oversample_coco_json(
    dataset_dir: str,
    train_json_rel: str = "annotations/instances_train.json",
    out_json_rel: str = "annotations/instances_train_oversampled.json",
    cap: int = 5,
):
    """
    Duplicate minority-class images in COCO train json.

    Strategy:
      - Compute per-class counts
      - class_weight = avg_count / class_count
      - image repeat k = min(cap, round(max(class_weight of its classes)))
      - Duplicate image entry with NEW image_id and duplicate annotations to point to that new id.
      - File names are unchanged (point to same image files).
    """
    train_json = os.path.join(dataset_dir, train_json_rel)
    coco = _read_coco(train_json)

    img_to_cls = _build_imgid_to_classes(coco)
    cls_cnt = _class_counts(coco)
    if not cls_cnt:
        raise RuntimeError("[OVERSAMPLE] No annotations found in train json.")

    total = sum(cls_cnt.values())
    avg = total / max(1, len(cls_cnt))
    cls_w = {c: max(1.0, avg / cnt) for c, cnt in cls_cnt.items()}  # inverse-frequency weighting

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    next_img_id = (max([im["id"] for im in coco["images"]]) if coco["images"] else 0) + 1
    next_ann_id = (max([a["id"] for a in coco["annotations"]]) if coco["annotations"] else 0) + 1

    new_images, new_anns = [], []

    for img_id, classes in img_to_cls.items():
        if not classes:
            continue
        w = max(cls_w[c] for c in classes)
        k = int(round(w))
        k = min(max(k, 1), cap)
        if k == 1:
            continue  # no duplication needed

        src_img = images_by_id[img_id]
        src_anns = anns_by_img.get(img_id, [])

        for _ in range(k - 1):
            dup_img = dict(src_img)
            dup_img["id"] = next_img_id
            new_images.append(dup_img)

            for a in src_anns:
                dup_ann = dict(a)
                dup_ann["id"] = next_ann_id
                dup_ann["image_id"] = next_img_id
                new_anns.append(dup_ann)
                next_ann_id += 1

            next_img_id += 1

    coco_os = {
        "images": coco["images"] + new_images,
        "annotations": coco["annotations"] + new_anns,
        "categories": coco["categories"],
    }
    out_json = os.path.join(dataset_dir, out_json_rel)
    _write_coco(coco_os, out_json)
    print(f"[OVERSAMPLE] class counts: {dict(cls_cnt)}")
    print(f"[OVERSAMPLE] +images={len(new_images)}  +anns={len(new_anns)}")
    print(f"[OVERSAMPLE] wrote: {out_json}")
    return out_json


# ---------------- RF-DETR training ----------------

def train_rfdetr_model(
    dataset_dir: str = "./data",      # ROOT containing train/, val/, annotations/
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda",
    cap: int = 5,
    replace_instances: bool = True,   # swap instances_train.json → oversampled json during training
    model_size: str = "small",        # "small" or "base"
):
    """
    Train RF-DETR on a COCO dataset with optional oversampling.
    The training loader always reads annotations/instances_train.json, so we temporarily overwrite it.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rfdetr_{timestamp}"

    # 1) Build oversampled train json
    oversampled_json = oversample_coco_json(
        dataset_dir=dataset_dir,
        train_json_rel="annotations/instances_train.json",
        out_json_rel="annotations/instances_train_oversampled.json",
        cap=cap,
    )

    # 2) (Optional) swap train json → oversampled
    ann_dir = os.path.join(dataset_dir, "annotations")
    orig_path = os.path.join(ann_dir, "instances_train.json")
    bak_path = os.path.join(ann_dir, "instances_train.json.bak")
    if replace_instances:
        if os.path.exists(bak_path):
            os.remove(bak_path)
        os.replace(orig_path, bak_path)           # move original → .bak
        shutil.copy2(oversampled_json, orig_path) # copy oversampled → default name

    # 3) Build model
    model = RFDETRSmall() if model_size == "small" else RFDETRBase()
    history = []
    def on_epoch_end(data): history.append(data)
    model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

    # 4) Train
    model.train(
        dataset_dir=dataset_dir,  # expects train/, val/, annotations/instances_train.json
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        grad_accum_steps=4,
        # amp=False,
        # num_workers=0
    )

    # 5) Restore original json
    if replace_instances and os.path.exists(bak_path):
        os.replace(bak_path, orig_path)

    return model, history, run_name
