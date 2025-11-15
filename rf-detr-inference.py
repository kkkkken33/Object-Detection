import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSmall, RFDETRLarge  # or RFDETRBase
import yaml

id2word = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

word_to_id = {
    "pillow": 0,
    "box": 1,
    "book": 2,
    "bottle": 3,
    "chair": 4,
    "mug": 5,
    "door": 6,
    "shelf": 7,
    "plate": 8,
    "sofa": 9
}

# Path to your trained RF-DETR checkpoint
best_ckpt = "./rf-detr-large.pth"

# Load the RF-DETR model (change to RFDETRBase if you're using the base model)
model = RFDETRLarge(pretrain_weights=best_ckpt)  # or RFDETRBase()

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
    for class_id, confidence, bbox in zip(predictions.class_id, predictions.confidence, predictions.xyxy):
        # Convert from [x_center, y_center, width, height] to [x_min, y_min, width, height]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Map category_id to category name using category_names from dataset.yaml
        try:
            class_id = word_to_id[id2word[class_id]]
        except:
            continue
        # Append to the output list
        out.append({
            "image_id": image_id,
            "category_id": class_id,  # Use category ID
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

    if not predictions:  # No detections found, log it
        print(f"[INFO] No detections for image: {img_path.name} (image_id: {image_id})")
        # Optionally, append to the output that no detections were found
        results_out.append({
            "image_id": image_id,
            "category_id": None,
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
