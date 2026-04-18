import os
import shutil
import random
import torch
import subprocess
from PIL import Image

DETECTOR_WEIGHTS = "yolov5/runs/train/final_scute_detector/weights/best.pt"
ALL_IMAGES_DIR   = "all_images"   
CROPS_DIR        = "reid_crops"    
TRAIN_RATIO      = 0.8
RANDOM_SEED      = 42
random.seed(RANDOM_SEED)

subprocess.run([
    "python", "yolov5/classify/train.py",
    "--model",   "yolov5l.pt",
    "--data",    "C:/Users/MICxN/SIYA/endsem/reid_data",
    "--epochs",  "40",
    "--img",     "224",
    "--batch",   "32",
    "--name",    "reid_classifier"
], check=True)

print("\nClassifier trained.")
print("Weights: yolov5/runs/train-cls/reid_classifier/weights/best.pt")