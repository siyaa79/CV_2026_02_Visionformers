import os
import cv2
import numpy as np
import torch

YOLO_MODEL_PATH = "./weights/best.pt"
IMG_DIR = "./data/reid"
OUT_DIR = "./data/clean_crops"

def resize_with_padding(img, size=224):
    h, w = img.shape[:2]

    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def crop_best_scute():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
    model.conf = 0.1

    if torch.cuda.is_available():
        model.to('cuda')

    total = 0
    missed = 0
    skipped_small = 0

    for img_file in os.listdir(IMG_DIR):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMG_DIR, img_file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        h_img, w_img, _ = image.shape

        results = model(img_path)
        boxes = results.xyxy[0]

        if len(boxes) == 0:
            missed += 1
            continue

        conf_scores = boxes[:, 4]
        best_idx = torch.argmax(conf_scores)

        if conf_scores[best_idx] < 0.1:
            missed += 1
            continue

        x1, y1, x2, y2 = boxes[best_idx][:4].int().tolist()

        padding = 0.2
        dx = int((x2 - x1) * padding)
        dy = int((y2 - y1) * padding)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w_img, x2 + dx)
        y2 = min(h_img, y2 + dy)

        if (x2 - x1) < 80 or (y2 - y1) < 80:
            skipped_small += 1
            continue

        aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
        if aspect_ratio < 0.3 or aspect_ratio > 3:
            continue

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            missed += 1
            continue

        crop_resized = resize_with_padding(crop, 224)

        croc_id = img_file.split("_")[0]

        class_folder = os.path.join(OUT_DIR, croc_id)
        os.makedirs(class_folder, exist_ok=True)

        base_name = os.path.splitext(img_file)[0]
        save_path = os.path.join(class_folder, f"{base_name}_best.jpg")

        cv2.imwrite(save_path, crop_resized)

        print(f"{img_file} → conf: {conf_scores[best_idx]:.3f}")

        total += 1

    print("\n==========================")
    print(f"✅ Total crops saved: {total}")
    print(f"Missed images: {missed}")
    print(f"Small crops skipped: {skipped_small}")
    print("==========================")


if __name__ == "__main__":
    print("Cropping best scutes...")
    crop_best_scute()
    print("Done.")