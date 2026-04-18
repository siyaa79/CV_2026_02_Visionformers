import os
import cv2
import numpy as np
from ultralytics import YOLO

YOLO_MODEL_PATH = "./weights/best.pt"
TEST_IMAGE_DIR = "./data/test_raw"
OUTPUT_DIR = "./data/test_crops"


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


def main():
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(YOLO_MODEL_PATH)

    total = 0
    fallback = 0

    for img_file in os.listdir(TEST_IMAGE_DIR):
        if not img_file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2

        # 🔥 YOLO with relaxed threshold
        results = model.predict(
            img_path,
            conf=0.2,   # allow more detections
            iou=0.5,
            max_det=5,   # allow multiple → we select best
            imgsz=1024
        )

        boxes = results[0].boxes

        best_box = None
        min_dist = float("inf")

        if boxes is not None and len(boxes) > 0:

            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)

                bx = (x1 + x2) // 2
                by = (y1 + y2) // 2

                # 🔥 distance from center
                dist = (bx - cx) ** 2 + (by - cy) ** 2

                if dist < min_dist:
                    min_dist = dist
                    best_box = (x1, y1, x2, y2)

        if best_box is None:
            # 🔥 fallback: center crop
            fallback += 1
            x1 = int(0.3 * w)
            x2 = int(0.7 * w)
            y1 = int(0.3 * h)
            y2 = int(0.7 * h)
        else:
            x1, y1, x2, y2 = best_box

        # 🔥 padding
        padding = 0.2
        dx = int((x2 - x1) * padding)
        dy = int((y2 - y1) * padding)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = resize_with_padding(crop, 224)

        save_path = os.path.join(OUTPUT_DIR, img_file)
        cv2.imwrite(save_path, crop)

        total += 1

    print(f"✅ Crops: {total}")
    print(f"⚠️ Fallback used: {fallback}")


if __name__ == "__main__":
    main()