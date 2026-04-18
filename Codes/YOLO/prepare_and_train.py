import os
import shutil
import random
import subprocess

ANNOTATED_IMAGES = "annotated/images"
ANNOTATED_LABELS = "annotated/labels"
DATASET_DIR      = "dataset"
TRAIN_RATIO      = 0.8
RANDOM_SEED      = 42

random.seed(RANDOM_SEED)

for split in ["train", "val"]:
    os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)

all_images = [f for f in os.listdir(ANNOTATED_IMAGES) if f.endswith('.jpg')]
random.shuffle(all_images)

split_idx  = int(len(all_images) * TRAIN_RATIO)
train_imgs = all_images[:split_idx]   
val_imgs   = all_images[split_idx:]  

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")

for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
    for img_name in img_list:
        label_name = img_name.replace(".jpg", ".txt")
        
        shutil.copy(
            f"{ANNOTATED_IMAGES}/{img_name}",
            f"{DATASET_DIR}/images/{split}/{img_name}"
        )
        
        label_src = f"{ANNOTATED_LABELS}/{label_name}"
        if os.path.exists(label_src):
            shutil.copy(label_src,
                        f"{DATASET_DIR}/labels/{split}/{label_name}")

print("Dataset split done.")

yaml_content = f"""path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val
nc: 1
names: ['scute']
"""
with open("scute.yaml", "w") as f:
    f.write(yaml_content)
print("scute.yaml created.")

if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone",
                    "https://github.com/ultralytics/yolov5"], check=True)
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"],
                   check=True)
    print("YOLOv5 cloned and installed.")

subprocess.run([
    "python", "yolov5/train.py",
    "--img",     "640",
    "--batch",   "16",
    "--epochs",  "50",
    "--data",    "scute.yaml",
    "--weights", "yolov5l.pt",
    "--freeze",  "10",
    "--name",    "scute_detector"
], check=True)

print("\nTraining complete. Weights at: yolov5/runs/train/scute_detector/weights/best.pt")