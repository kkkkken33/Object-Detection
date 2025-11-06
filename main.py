# main.py — automatic train/resume/evaluate pipeline

import os
import json
from ultralytics import YOLO
from train import train_yolo_model


# Path to your last best model
# best_ckpt = "./runs/train_20251103_233323/weights/last.pt"
best_ckpt = ""  
data_yaml = "./data/dataset.yaml"

def main():
    # --------------------------------------------------
    # 1️⃣ If a trained model exists → resume from it
    # --------------------------------------------------
    if os.path.exists(best_ckpt):
        print(f"[INFO] Found existing checkpoint: {best_ckpt}")
        model = YOLO(best_ckpt)
        print("[INFO] Resuming fine-tuning for a few more epochs...")
        # Resume for extra epochs to refine model
        model.train(
            data="./data/dataset.yaml",
            resume=True,
            epochs=30,
            patience=10,
            device=0,
        )
    else:
        print("[INFO] No previous checkpoint found. Starting new training...")
        model = train_yolo_model(epochs=100, batch_size=16, img_size=1024, lr0=1e-3)

    # --------------------------------------------------
    # 2️⃣ Evaluate after training
    # --------------------------------------------------
    print("\n[INFO] Evaluating model on validation set...")
    if os.path.exists(best_ckpt):
        model = YOLO(best_ckpt)
        print(f"[INFO] Evaluating model from checkpoint: {best_ckpt}")
    else:
        print("[WARN] No trained model found. Using default pretrained model for evaluation.")
        model = YOLO('yolov8n.pt')
    results = model.val(data=data_yaml, save_json=True)

    # Display all key metrics
    print(
        f"\n✅ Validation Results:\n"
        f"mAP50-95: {results.box.map:.4f}\n"
        f"mAP50:    {results.box.map50:.4f}\n"
        f"Precision:{results.box.mp:.4f}\n"
        f"Recall:   {results.box.mr:.4f}\n"
    )

    # Save to file for later review
    with open("val_results.txt", "w") as f:
        json.dump({
            "mAP50-95": results.box.map,
            "mAP50": results.box.map50,
            "Precision": results.box.mp,
            "Recall": results.box.mr
        }, f, indent=2)
    print("[INFO] Validation metrics saved to val_results.txt")

if __name__ == "__main__":
    main()
