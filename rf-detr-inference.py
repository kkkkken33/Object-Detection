import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSmall  # or RFDETRBase
import yaml

# Path to your trained RF-DETR checkpoint
best_ckpt = "./runs/train_20251108_0054402/weights/best.pt"

# Load the RF-DETR model (change to RFDETRBase if you're using the base model)
model = RFDETRSmall()  # or RFDETRBase()
model.load_state_dict(torch.load(best_ckpt))  # Load the saved state_dict
model.eval()  # Set the model to evaluation mode

# Load categories from dataset.yaml
def load_categories_from_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # Return the class names (assuming the 'names' field contains the categories)
    return data.get("names", [])

# Path to the dataset.yaml
data_yaml = "./data/dataset.yaml"

# Load category names from dataset.yaml
category_names = load_categories_from_yaml(data_yaml)

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
    for class_id, confidence, bbox in zip(predictions.class_id, predictions.confidence, predictions.xywh):
        # Convert from [x_center, y_center, width, height] to [x_min, y_min, width, height]
        x_min = bbox[0] - bbox[2] / 2
        y_min = bbox[1] - bbox[3] / 2
        width = bbox[2]
        height = bbox[3]

        # Map category_id to category name using category_names from dataset.yaml
        category_name = category_names[class_id] if class_id < len(category_names) else "unknown"

        # Append to the output list
        out.append({
            "image_id": image_id,
            "category_id": class_id,  # Use category ID
            "category_name": category_name,  # Optional: You can also include the class name
            "bbox": [x_min, y_min, width, height],  # [x_min, y_min, width, height]
            "score": confidence
        })
    return out

# Perform inference on all images in the source folder
results_out = []
for img_path in tqdm(sorted(Path(source).glob("*.*"))):  # Iterate through all image files in the directory
    image_id = None
    stem = img_path.stem
    try:
        image_id = int(stem)  # Try to extract image_id from filename
    except:
        image_id = stem  # Fallback to the filename if it can't be converted

    img = Image.open(img_path).convert("RGB")
    
    # Perform inference with the model
    predictions = model.predict(img, threshold=0.5)  # Perform inference with a confidence threshold

    if not predictions.predictions:  # No detections found, log it
        print(f"[INFO] No detections for image: {img_path.name} (image_id: {image_id})")
        # Optionally, append to the output that no detections were found
        results_out.append({
            "image_id": image_id,
            "category_id": None,
            "category_name": None,
            "bbox": None,
            "score": None
        })
        continue  # Skip further processing for this image

    # Convert RF-DETR predictions to COCO format
    coco_preds = _to_coco_format(predictions, image_id)
    results_out.extend(coco_preds)

# Save results to a JSON file
timestamp = time.strftime('%Y%m%d_%H%M%S')
out_file = f"inference_results_rfdetr_{timestamp}.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results_out, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results_out)} detections to {out_file}")
