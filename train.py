# train.py (improved)
import os, shutil, torch
from datetime import datetime
from ultralytics import YOLO

def train_yolo_model(epochs=100, batch_size=16, lr0=1e-3, data_yaml_path="./data/dataset.yaml", base_dir=".", model_save_dir="./runs/saved_models"):
    """Enhanced YOLO training with better augmentation, optimizer, and stability."""
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

    # ---------------- Train configuration ----------------
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        # imgsz=img_size,
        device=device,
        project=os.path.join(base_dir, "runs"),
        name=run_name,
        patience=20,               # wait longer before early stop
        save=True,
        save_period=5,
        plots=True,
        # --- Optimizer / Scheduler ---
        optimizer="AdamW",
        lr0=lr0,                   # initial LR
        lrf=0.1,                   # cosine LR end
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        cos_lr=True,
        # --- Advanced augmentations ---
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.8, shear=0.1,
        flipud=0.0, fliplr=0.5,
        erasing=0.3,
        close_mosaic=10,           # close mosaic last 10 epochs
        # --- Regularization & speed ---
        amp=True,
        cache=True,
        workers=4,
        seed=42,
        val=True                   # validate each epoch
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
