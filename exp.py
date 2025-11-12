import os
import json
from ultralytics import YOLO

model = YOLO("yolo12m.pt")
print(model.model)