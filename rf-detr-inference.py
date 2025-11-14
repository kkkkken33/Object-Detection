from rfdetr import RFDETRBase, RFDETRSmall
from tqdm import tqdm
from pathlib import Path
import json
import os
import time
import torch
from PIL import Image

# Path to the best model checkpoint
best_ckpt = "./runs/train_20251108_0054402/weights/best.pt"

# Load RF-DETR model (change to small or base model depending on your setup)
model = RFDETRSmall.load(best_ckpt)  # or RFDETRBase.load(best_ckpt)

# Define path to the directory containing images and videos for inference
source = "./data/images/test"

# Function to perform inference
def perform_inference(image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Perform inference using the RF-DETR model (you can set a threshold to filter out low-confidence detections)
    results = model.predict(img, threshold=0.5)  # Set the threshold to 0.5 for filtering low confidence

    # Parse the results
    out = []
    for result in results:
        # result should contain bounding boxes, category ids, and scores
        xyxy = result["bbox"]  # [x, y, width, height] (format: xyxy or [x1, y1, x2, y2])
        category_id = result["category_id"]  # The class id of the predicted object
        score = result["score"]  # Confidence score

        # Calculate width and height from the bbox (if in [x, y, w, h] format)
        x, y, w, h = xyxy
        out.append({
            "image_id": image_path,  # Use the image path or file name as the ID
            "category_id": category_id,
            "bbox": [x, y, w, h],  # Use [x, y, width, height]
            "score": score
        })

    return out

# Process the source directory for inference
out = []
for image_path in tqdm(Path(source).glob("*.*")):  # Iterate through all image files in the directory
    result = perform_inference(image_path)
    out.extend(result)

# Save results to a JSON file
timestamp = time.strftime('%Y%m%d_%H%M%S')
output_filename = f"inference_results_{timestamp}.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved {len(out)} detections to {output_filename}")
