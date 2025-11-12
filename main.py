# main.py — RF-DETR automatic train/resume/evaluate pipeline

import os
import json
from rfdetr import RFDETRBase
from train import train_rfdetr_model  # Import RF-DETR training function from train.py

# Path to your last best model (RF-DETR checkpoint)
best_ckpt = ""  # Example: "./runs/train_20251103_233323/weights/best.pt"
data_yaml = "./data/dataset.yaml"  # Path to your dataset YAML

def main(is_freeze=False, is_yolo=True):
    # # --------------------------------------------------
    # # 1️⃣ If a trained model exists → resume from it
    # # --------------------------------------------------
    if is_yolo == False:
        train_rfdetr_model()
    if os.path.exists(best_ckpt):
        print(f"[INFO] Found existing checkpoint: {best_ckpt}")
        # Load RF-DETR model checkpoint
        model = RFDETRBase.load(best_ckpt)  # RF-DETR method to load pre-trained models
        print("[INFO] Resuming fine-tuning for a few more epochs...")
        # Resume fine-tuning (adjust epochs as needed)
        model.train(
            dataset_dir="./data",   # COCO-style dataset path
            epochs=30,              # Additional epochs for fine-tuning
            batch_size=16,
            lr=1e-4,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        )
    elif is_freeze == False:
        print("[INFO] No previous checkpoint found. Starting new training...")
        # Train RF-DETR from scratch
        model, history, run_name = train_rfdetr_model(
            dataset_dir="./data",   # Path to the COCO dataset
            epochs=100,             # Training epochs
            batch_size=16,
            lr=1e-4,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        )

    # --------------------------------------------------
    # 2️⃣ Evaluate after training
    # --------------------------------------------------
    print("\n[INFO] Evaluating model on validation set...")
    if os.path.exists(best_ckpt):
        model = RFDETRBase.load(best_ckpt)  # Load the best checkpoint
        print(f"[INFO] Evaluating model from checkpoint: {best_ckpt}")
    else:
        print("[INFO] No trained model found. Using default pre-trained model for evaluation.")
        model = RFDETRBase()  # Load the base model if no checkpoint found
    
    # Run evaluation and save results
    results = model.val(
        dataset_dir="./data",     # Path to the COCO dataset
        data_yaml=data_yaml,     # Your dataset YAML file
        save_json=True
    )

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
    main(is_freeze=False, is_yolo=False)
