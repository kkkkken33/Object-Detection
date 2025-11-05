# train.py (Option B: auto-oversample without manually editing your YAML)
import os, shutil, torch, glob, collections
from datetime import datetime
from ultralytics import YOLO

# If you don't have PyYAML, install it: pip install pyyaml
try:
    import yaml
except Exception:
    yaml = None

def _resolve_path(root, p):
    """Return absolute path for `p` which can be absolute, relative to root, or a txt file."""
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    if root and os.path.isabs(root):
        return os.path.normpath(os.path.join(root, p))
    return os.path.abspath(p)

def _load_dataset_yaml(data_yaml_path):
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Normalize fields
    root = cfg.get("path", None)
    train = cfg.get("train")
    val = cfg.get("val")
    names = cfg.get("names")
    return cfg, root, train, val, names

def _make_oversampled_txt(root, train_spec, cap=10):
    """
    Build an oversampled train.txt by duplicating images containing minority classes.
    - root: dataset root from YAML's `path:` (can be None)
    - train_spec: either a folder or an existing txt
    Returns: absolute path to train_oversampled.txt
    """
    # Resolve train source
    train_path = _resolve_path(root, train_spec)
    assert os.path.exists(train_path), f"Train path not found: {train_path}"

    # Collect (image -> classes) mapping
    img_to_classes = {}
    cls_counts = collections.Counter()

    if os.path.isdir(train_path):
        # Expect YOLO layout: images/train + labels/train
        # infer labels dir
        if "images" + os.sep in train_path:
            labels_dir = train_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        else:
            # try sibling labels/train
            labels_dir = train_path.replace("images", "labels")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"Labels folder not found for oversampling: {labels_dir}")

        # iterate labels
        for lb in glob.glob(os.path.join(labels_dir, "*.txt")):
            # map to image path (jpg/png)
            base = os.path.splitext(os.path.basename(lb))[0]
            img_jpg = os.path.join(train_path, base + ".jpg")
            img_png = os.path.join(train_path, base + ".png")
            img = img_jpg if os.path.exists(img_jpg) else img_png if os.path.exists(img_png) else None
            if img is None:
                continue

            classes = set()
            with open(lb, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    c = int(parts[0])
                    classes.add(c)
                    cls_counts[c] += 1
            if classes:
                img_to_classes[img] = classes

    elif os.path.isfile(train_path) and train_path.endswith(".txt"):
        # If the YAML already points to a train.txt, we’ll read the list and
        # locate corresponding label files to count classes.
        lines = [l.strip() for l in open(train_path) if l.strip()]
        for img in lines:
            # guess label path
            lb = img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            lb = os.path.splitext(lb)[0] + ".txt"
            if not os.path.exists(lb):
                continue
            classes = set()
            with open(lb, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    c = int(parts[0])
                    classes.add(c)
                    cls_counts[c] += 1
            if classes:
                img_to_classes[img] = classes
    else:
        raise ValueError(f"Unsupported train spec: {train_spec}")

    if not cls_counts:
        raise RuntimeError("No labels found while building oversampled list. Check dataset paths.")

    # Compute inverse-frequency weights
    total = sum(cls_counts.values())
    avg = total / max(1, len(cls_counts))
    cls_weight = {c: max(1.0, avg / cnt) for c, cnt in cls_counts.items()}

    # Per-image weight = max class weight in the image
    img_weight = {img: max(cls_weight[c] for c in classes) for img, classes in img_to_classes.items()}

    # Write oversampled txt alongside the train source (or under root)
    out_dir = os.path.dirname(train_path) if os.path.isdir(train_path) else (root or os.path.dirname(train_path))
    out_txt = os.path.join(out_dir, "train_oversampled.txt")
    lines_out = 0
    with open(out_txt, "w") as out:
        for img, w in img_weight.items():
            k = int(round(w))
            k = min(max(k, 1), cap)  # cap duplication
            for _ in range(k):
                out.write(img + "\n")
                lines_out += 1

    print(f"[OVERSAMPLE] class counts: {dict(cls_counts)}")
    print(f"[OVERSAMPLE] wrote: {out_txt} (lines={lines_out})")
    return out_txt

def _write_temp_yaml(original_yaml_path, root, val_spec, names, oversampled_txt):
    """Create a sibling YAML that points train to the oversampled txt; return its path."""
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    with open(original_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Preserve everything but override 'train' only
    cfg["train"] = oversampled_txt
    # Keep root, val, names as-is; ensure absolute/relative work:
    # (Ultralytics can resolve absolute txt directly; 'path' can stay)
    tmp_yaml = os.path.join(os.path.dirname(original_yaml_path), "dataset_oversampled.yaml")
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[OVERSAMPLE] wrote YAML: {tmp_yaml}")
    return tmp_yaml

def train_yolo_model(
    epochs=100, batch_size=16, img_size=1024, lr0=1e-3,
    data_yaml_path="./data/dataset.yaml", base_dir=".",
    model_save_dir="./runs/saved_models"
):
    """Enhanced YOLO training with class-imbalance oversampling (Option B) and stable settings."""
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs"), exist_ok=True)

    device = '0' if torch.cuda.is_available() else 'cpu'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"

    # ---------------- Load pretrained model ----------------
    try:
        model = YOLO('yolo12n.pt')
        model_type = 'yolo12n'
    except Exception:
        model = YOLO('yolov8n.pt')
        model_type = 'yolov8n'
    print(f"[INFO] Using model: {model_type} on device={device}")

    # ---------------- Oversample (Option B) ----------------
    cfg, root, train_spec, val_spec, names = _load_dataset_yaml(data_yaml_path)
    oversampled_txt = _make_oversampled_txt(root, train_spec, cap=10)
    oversampled_yaml = _write_temp_yaml(data_yaml_path, root, val_spec, names, oversampled_txt)

    # ---------------- Train configuration ----------------
    # Light augmentations only (mosaic/mixup/copy_paste OFF)
    # Add focal loss knobs for imbalance (supported in 8.3.x; if you see SyntaxError, remove fl_gamma)
    results = model.train(
        data=oversampled_yaml,      # ← train now points to oversampled txt
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=os.path.join(base_dir, "runs"),
        name=run_name,
        patience=20,
        save=True,
        save_period=5,
        plots=True,

        # --- Optimizer / Scheduler ---
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.1,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        cos_lr=True,

        # --- Light augs only ---
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=2.0, translate=0.05, scale=0.5, shear=0.05,
        fliplr=0.5, flipud=0.0,

        # --- Imbalance-friendly loss tweaks ---
        cls=1.0,          # give a bit more weight to classification

        # --- Regularization & speed ---
        amp=True,
        cache=True,
        workers=4,
        seed=42,
        val=True
    )

    # ---------------- Save best model ----------------
    model_save_path = os.path.join(model_save_dir, f"{model_type}_{timestamp}.pt")
    best_model_path = os.path.join(base_dir, "runs", run_name, "weights", "best.pt")

    try:
        model.model.save(model_save_path)
    except AttributeError:
        try:
            model.save(model_save_path)
        except Exception:
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_save_path)
    print(f"[INFO] ✅ Model saved to {model_save_path}")
    return model
