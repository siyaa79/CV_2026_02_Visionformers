import os
import torch
import torch.nn.functional as F
import csv
from PIL import Image
import torchvision.transforms as transforms
import sys

DETECTOR_WEIGHTS   = "yolov5/runs/train/final_scute_detector/weights/best.pt"
CLASSIFIER_WEIGHTS = "yolov5/runs/train-cls/reid_classifier3/weights/best.pt"
INPUT_DIR          = "datasets"
OUTPUT_CSV         = "predictions.csv"
CONF_THRESHOLD     = 0.70
CLASSES            = ["107", "108", "109", "110", "111", "112"]
DEVICE             = "0" if torch.cuda.is_available() else "cpu"      
DEVICE_PT          = "cuda" if torch.cuda.is_available() else "cpu"   

print(f"Using device: {DEVICE_PT}")

detector      = torch.hub.load('ultralytics/yolov5', 'custom',
                                path=DETECTOR_WEIGHTS, device=DEVICE)
detector.conf = 0.10

sys.path.insert(0, "yolov5")
ckpt       = torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE_PT)
classifier = ckpt['model'].float().eval().to(DEVICE_PT)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_scute_crop(img_path):
    results = detector(img_path)
    boxes   = results.xyxy[0]
    img     = Image.open(img_path).convert("RGB")

    if len(boxes) == 0:
        return img, "no_detection"

    best            = boxes[boxes[:, 4].argmax()]
    x1, y1, x2, y2 = map(int, best[:4])
    return img.crop((x1, y1, x2, y2)), "detected"


def predict_class(crop):
    tensor = transform(crop).unsqueeze(0).to(DEVICE_PT)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = F.softmax(logits, dim=1)

    max_conf, pred_idx = probs.max(1)
    return CLASSES[pred_idx.item()], max_conf.item(), probs[0].tolist()

test_images = sorted([f for f in os.listdir(INPUT_DIR)
                      if f.lower().endswith('.jpg')])
results = []

for img_name in test_images:
    img_path = f"{INPUT_DIR}/{img_name}"
    crop, detection_status = get_scute_crop(img_path)

    if detection_status == "no_detection":
        pred_class = "UNKNOWN"
        confidence = 0.0
        all_probs  = [0.0] * len(CLASSES)
        reason     = "no_scute_detected"
    else:
        pred_class, confidence, all_probs = predict_class(crop)

        if confidence < CONF_THRESHOLD:
            reason     = f"low_confidence_{confidence:.2f}"
            pred_class = "UNKNOWN"
        else:
            reason = "identified"

    results.append({
        "image":      img_name,
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "reason":     reason,
        **{f"prob_{c}": round(p, 4)
           for c, p in zip(CLASSES, all_probs)}
    })

    print(f"{img_name:30s} → {pred_class:10s} "
          f"(conf: {confidence:.2f}) [{reason}]")

fieldnames = ["image", "prediction", "confidence", "reason"] + \
             [f"prob_{c}" for c in CLASSES]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

known   = sum(1 for r in results if r["prediction"] != "UNKNOWN")
unknown = sum(1 for r in results if r["prediction"] == "UNKNOWN")

print(f"\n{'─'*50}")
print(f"Total   : {len(results)}")
print(f"Known   : {known}")
print(f"Unknown : {unknown}")
print(f"Saved   : {OUTPUT_CSV}")