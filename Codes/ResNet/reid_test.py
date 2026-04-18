import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import csv

MODEL_PATH = "./weights/reid_resnet50.pth"
TRAIN_DIR = "./data/clean_crops"
TEST_DIR = "./data/test_crops"
OUTPUT_CSV = "./outputs/reid_results.csv"

class ReIDModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return nn.functional.normalize(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ReIDModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img)

print("🔵 Building embedding database...")
database = []

for label in os.listdir(TRAIN_DIR):
    label_path = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for img in os.listdir(label_path):
        database.append((label, get_embedding(os.path.join(label_path, img))))

print(f"✅ Database size: {len(database)}")

print("🔵 Running ReID testing...")
results = []

for img in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img)
    emb_test = get_embedding(img_path)

    best_dist = float("inf")
    best_label = None

    for label, emb_db in database:
        dist = torch.norm(emb_test - emb_db).item()
        if dist < best_dist:
            best_dist = dist
            best_label = label

    confidence = 1 / (1 + best_dist)
    pred = "unknown" if best_dist > 1.2 else best_label

    try:
        gt_id = img.split("_")[1]
    except:
        gt_id = "NA"

    results.append([img, gt_id, pred, round(confidence, 4)])
    print(f"{img} → Pred: {pred}, Dist: {best_dist:.3f}")

os.makedirs("./outputs", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Test ID", "Predicted", "Conf"])
    writer.writerows(results)

print(f"✅ Results saved to: {OUTPUT_CSV}")