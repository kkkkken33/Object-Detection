import json, sys, os

def patch(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # Ensure required top-level keys exist
    d.setdefault("info", {"description": "patched", "version": "1.0"})
    d.setdefault("licenses", [])
    # Ensure required lists exist (and are lists)
    for k in ("images", "annotations", "categories"):
        if k not in d or not isinstance(d[k], list):
            raise ValueError(f"COCO file '{path}' is missing required list '{k}'")
    # (Optional) ensure ints
    for im in d["images"]:
        if "id" in im: im["id"] = int(im["id"])
    for ann in d["annotations"]:
        if "id" in ann: ann["id"] = int(ann["id"])
        if "image_id" in ann: ann["image_id"] = int(ann["image_id"])
        if "category_id" in ann: ann["category_id"] = int(ann["category_id"])
    # Write back
    tmp = path + ".fixed"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"Patched -> {tmp}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_coco_json.py path/to/instances_val.json")
        sys.exit(1)
    patch(sys.argv[1])

"""
Usage:
python fix_coco_json.py ./data/images/valid/_annotations.coco.json
# (optionally)
python fix_coco_json.py path/to/instances_train.json
"""
