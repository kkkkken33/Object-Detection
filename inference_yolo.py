# ...existing code...
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import json
import os
import time

best_ckpt = "./runs/train_20251108_0054402/weights/best.pt"

# Load a pretrained YOLO12n model
model = YOLO(best_ckpt)

# Define path to directory containing images and videos for inference
source = "./data/images/test"

# Run inference on the source (generator)
results = model(source, stream=True)  # generator of Results objects

# for result in results:
#     print(result.to_json())
#     result.show()
#     break

out = []
# Process results generator
for result in tqdm(results):
    # 获取图片 id（优先用文件名能转为 int 的值）
    path = getattr(result, "path", None) or getattr(result, "orig_path", None) or None
    if path:
        stem = Path(path).stem
        try:
            image_id = int(stem)
        except Exception:
            image_id = stem
    else:
        # fallback: 使用 result.img 的索引或 None
        image_id = getattr(result, "id", None)

    boxes = getattr(result, "boxes", None)  # Boxes object
    if not boxes:
        continue

    # 使用 xyxy（归一化中心x,中心y,width,height，相对于原始图像）
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
    conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []

    for i in range(len(xyxy)):
        x, y, ex, ey = map(float, xyxy[i])
        w = ex - x
        h = ey - y
        category_id = int(cls[i]) if len(cls) > i else None
        score = float(conf[i]) if len(conf) > i else None

        out.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],  # normalized [x, y, width, height] (center-based as xyxy)
            "score": score
        })
        # print(image_id, category_id, [cx, cy, w, h], score)
        # print()
    # break

# Save results to a JSON file
with open(f"inference_results_{time.strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved {len(out)} detections to inference_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
