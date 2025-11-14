import os
import json
import glob
from PIL import Image
import yaml

# Paths
images_dir = './data/images/train'  # Path to your training images folder
labels_dir = './data/labels/train'  # Path to your training labels folder
output_json_path = './data/annotations/instances_train.json'  # Output path for COCO JSON
# === Load category names from YAML ===
yaml_path = './data/dataset.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
names = data['names']

categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(names)]

def convert_yolo_to_coco(images_dir, labels_dir, categories):
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    # Prepare category mapping
    category_map = {category['name']: category['id'] for category in categories}

    # Process each image
    for img_path in glob.glob(os.path.join(images_dir, "*.jpg")):  # Assuming .jpg, change if necessary
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        width, height = img.size
        
        # Create image entry
        image_entry = {
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        }
        images.append(image_entry)

        # Process annotations
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Convert YOLO to COCO format (bounding box in pixels)
                    x_min = (x_center - width / 2) * width
                    y_min = (y_center - height / 2) * height
                    bbox = [x_min, y_min, width * width, height * height]
                    
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    }
                    annotations.append(annotation_entry)
                    annotation_id += 1

        image_id += 1

    # Create final COCO JSON structure
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Converted YOLO annotations to COCO format. Saved to {output_json_path}")

# Run the conversion
convert_yolo_to_coco(images_dir, labels_dir, categories)
