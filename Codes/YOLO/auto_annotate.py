import os
import torch
from PIL import Image
import shutil

WEIGHTS        = "yolov5/runs/train/scute_detector/weights/best.pt"
UNANNOTATED    = "unannotated"
OUTPUT_IMAGES  = "auto_annotations/images"
OUTPUT_LABELS  = "auto_annotations/labels"
CONF_THRESHOLD = 0.25

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

model      = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS)
model.conf = CONF_THRESHOLD

all_images = [f for f in os.listdir(UNANNOTATED) if f.endswith('.jpg')]
no_detection = []

for img_name in all_images:
    img_path = f"{UNANNOTATED}/{img_name}"
    
    results  = model(img_path)
    boxes    = results.xyxy[0] 
    
    if len(boxes) == 0:
        no_detection.append(img_name)
        continue
    
    img = Image.open(img_path)
    W, H = img.size
    
    best     = boxes[boxes[:, 4].argmax()]
    x1, y1, x2, y2 = best[:4].tolist()
    
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    
    shutil.copy(img_path, f"{OUTPUT_IMAGES}/{img_name}")
    
    label_name = img_name.replace(".jpg", ".txt")
    with open(f"{OUTPUT_LABELS}/{label_name}", "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

print(f"Auto-annotated : {len(all_images) - len(no_detection)}")
print(f"No detection   : {len(no_detection)}")
if no_detection:
    print("Failed images:", no_detection[:10], "...")