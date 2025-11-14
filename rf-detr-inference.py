import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSmall  # or RFDETRBase

# Path to your trained RF-DETR checkpoint
best_ckpt = "./runs/train_20251108_0054402/weights/best.pt"

# Load the RF-DETR model (change to RFDETRBase if you're using the base model)
model = RFDETRSmall.load(best_ckpt)  # or RFDETRBase.load(best_ckpt)

# Directory containing images for inference
source = "./data/images/test"

# Function to convert RF-DETR predictions into COCO-style output
def _to_coco_format(predictions, image_id):
    """
    Convert RF-DETR predictions to COCO-style format:
    {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height],
        "score": score
    }
    """
    out = []
    for pred in predictions.predictions:
        # Extract the relevant information
        category_id = int(pred.class_id)  # Get the class ID
        bbox = pred.xywh  # Get [x_center, y_center, width, height]
        score = float(pred.confidence)  # Get confidence score
        
        # Convert from [x_center, y_center, width, height] to [x_min, y_min, width, height]
        x_min = bbox[0] - bbox[2] / 2
        y_min = bbox[1] - bbox[3] / 2
        width = bbox[2]
        height = bbox[3]

        # Append the result in the required format
        out.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, width, height],  # [x_min, y_min, width, height]
            "score": score
        })
    return out

# Process the source directory and perform inference
results_out = []
for img_path in tqdm(sorted(Path(source).glob("*.*"))):  # Iterate through all image files in the directory
    image_id = None
    stem = img_path.stem
    try:
        image_id = int(stem)  # Try to extract image_id from filename
    except:
        image_id = stem  # Fallback to the filename if it can't be converted

    img = Image.open(img_path).convert("RGB")
    predictions = model.infer(img, confidence=0.5)[0]  # Perform inference with a confidence threshold

    if not predictions.predictions:  # No detections found, skip the image
        continue

    # Convert RF-DETR predictions to COCO format
    coco_preds = _to_coco_format(predictions, image_id)
    results_out.extend(coco_preds)

# Save results to a JSON file
timestamp = time.strftime('%Y%m%d_%H%M%S')
out_file = f"inference_results_rfdetr_{timestamp}.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results_out, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results_out)} detections to {out_file}")
