import os
import random
import shutil

SRC_DIR = "./data/raw"
BASE_DIR = "./data"

train_img = os.path.join(BASE_DIR, "images/train")
val_img   = os.path.join(BASE_DIR, "images/val")
train_lbl = os.path.join(BASE_DIR, "labels/train")
val_lbl   = os.path.join(BASE_DIR, "labels/val")

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = [f for f in os.listdir(SRC_DIR) if f.endswith(".jpg")]

random.seed(42)
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_files = images[:split_idx]
val_files   = images[split_idx:]

def move(files, img_dst, lbl_dst):
    for img in files:
        txt = img.replace(".jpg", ".txt")

        src_img = os.path.join(SRC_DIR, img)
        src_txt = os.path.join(SRC_DIR, txt)

        if not os.path.exists(src_txt):
            continue

        shutil.copy(src_img, os.path.join(img_dst, img))
        shutil.copy(src_txt, os.path.join(lbl_dst, txt))

move(train_files, train_img, train_lbl)
move(val_files, val_img, val_lbl)

print("✅ Dataset split complete!")