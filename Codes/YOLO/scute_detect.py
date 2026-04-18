import os
import shutil
import random
import subprocess
import yaml

FINAL_DATASET = "final_dataset"  
TRAIN_RATIO   = 0.8
RANDOM_SEED   = 42
random.seed(RANDOM_SEED)

for split in ["train", "val"]:
    os.makedirs(f"detector_data/images/{split}", exist_ok=True)
    os.makedirs(f"detector_data/labels/{split}", exist_ok=True)

all_images = [f for f in os.listdir(f"{FINAL_DATASET}/images") 
              if f.endswith('.jpg')]
random.shuffle(all_images)

split_idx  = int(len(all_images) * TRAIN_RATIO)
train_imgs = all_images[:split_idx]
val_imgs   = all_images[split_idx:]

print(f"Total : {len(all_images)}")
print(f"Train : {len(train_imgs)}")
print(f"Val   : {len(val_imgs)}")

for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
    for img_name in img_list:
        label_name = img_name.replace(".jpg", ".txt")

        shutil.copy(
            f"{FINAL_DATASET}/images/{img_name}",
            f"detector_data/images/{split}/{img_name}"
        )
        label_src = f"{FINAL_DATASET}/labels/{label_name}"
        if os.path.exists(label_src):
            shutil.copy(
                label_src,
                f"detector_data/labels/{split}/{label_name}"
            )

print("Dataset split done.")

config = {
    "path": os.path.abspath("detector_data"),
    "train": "images/train",
    "val":   "images/val",
    "nc":    1,
    "names": ["scute"]
}
with open("scute.yaml", "w") as f:
    yaml.dump(config, f)
print("scute.yaml created.")

subprocess.run([
    "python", "yolov5/train.py",
    "--img",     "640",
    "--batch",   "16",
    "--epochs",  "80",
    "--data",    "scute.yaml",
    "--weights", "yolov5l.pt",
    "--freeze",  "10",
    "--name",    "final_scute_detector",
    "--patience", "20"        
], check=True)

print("\nDetector trained.")
print("Weights: yolov5/runs/train/final_scute_detector/weights/best.pt")