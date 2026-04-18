import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as transforms
import csv
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CLASSIFIER_WEIGHTS = "yolov5/runs/train-cls/reid_classifier3/weights/best.pt"
DETECTOR_WEIGHTS   = "yolov5/runs/train/final_scute_detector/weights/best.pt"
CROPS_DIR          = "reid_crops"
DATASET3_DIR       = "datasets"
CLASSES            = ["107", "108", "109", "110", "111", "112"]
MAX_PER_CLASS      = 150
NN_THRESHOLDS      = np.arange(0.3, 1.0, 0.05)
OUTPUT_DIR         = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = torch.hub.load('ultralytics/yolov5', 'custom',
                          path=DETECTOR_WEIGHTS).to(device)
detector.conf = 0.10

classifier = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=CLASSIFIER_WEIGHTS).to(device)
classifier.eval()

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
        return img, "fallback"

    best = boxes[boxes[:, 4].argmax()]
    x1, y1, x2, y2 = map(int, best[:4])
    return img.crop((x1, y1, x2, y2)), "detected"


def get_embedding(crop):
    tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        features = classifier.model.model[:-1](tensor)

        if features.dim() == 4:
            features = features.mean(dim=[2, 3])

        features = features.flatten()
        features = F.normalize(features, dim=0)

    return features.cpu().numpy()

gallery_embeddings = []
gallery_labels     = []
class_centroids    = {}

for class_id in CLASSES:
    class_dir = f"{CROPS_DIR}/{class_id}"
    if not os.path.exists(class_dir):
        print(f"WARNING: {class_dir} not found")
        continue

    images = [f for f in os.listdir(class_dir)
              if f.endswith('.jpg')][:MAX_PER_CLASS]

    print(f"Class {class_id}: {len(images)} images")

    class_embs = []
    for img_name in images:
        try:
            crop, _ = get_scute_crop(f"{class_dir}/{img_name}")
            emb     = get_embedding(crop)

            gallery_embeddings.append(emb)
            gallery_labels.append(class_id)
            class_embs.append(emb)

        except Exception as e:
            print(f"Skipped {img_name}: {e}")

    if class_embs:
        centroid = np.mean(class_embs, axis=0)
        class_centroids[class_id] = centroid / np.linalg.norm(centroid)

gallery_embeddings = np.array(gallery_embeddings)

print(f"Gallery size: {len(gallery_embeddings)}")
print(f"Embedding dim: {gallery_embeddings.shape[1]}")


MAX_TSNE = 600
if len(gallery_embeddings) > MAX_TSNE:
    idx = np.random.choice(len(gallery_embeddings), MAX_TSNE, replace=False)
    tsne_embs   = gallery_embeddings[idx]
    tsne_labels = [gallery_labels[i] for i in idx]
else:
    tsne_embs   = gallery_embeddings
    tsne_labels = gallery_labels

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
reduced = tsne.fit_transform(tsne_embs)

plt.figure(figsize=(10, 8))
for i, class_id in enumerate(CLASSES):
    mask = [l == class_id for l in tsne_labels]
    pts  = reduced[np.array(mask)]
    if len(pts):
        plt.scatter(pts[:,0], pts[:,1], label=class_id, alpha=0.6, s=25)

plt.legend()
plt.title("t-SNE Embeddings")
plt.savefig(f"{OUTPUT_DIR}/tsne_embeddings.png", dpi=150)
plt.show()


test_images = sorted([f for f in os.listdir(DATASET3_DIR)
                      if f.endswith('.jpg')])

test_embeddings = []
test_names = []

for img_name in test_images:
    try:
        crop, _ = get_scute_crop(f"{DATASET3_DIR}/{img_name}")
        emb     = get_embedding(crop)

        test_embeddings.append(emb)
        test_names.append(img_name)

    except Exception as e:
        print(f"Skipped {img_name}: {e}")

test_embeddings = np.array(test_embeddings)


def predict_1nn(test_emb, gallery_embs, gallery_lbls, threshold):
    sims = gallery_embs @ test_emb
    idx  = np.argmax(sims)
    sim  = sims[idx]
    pred = gallery_lbls[idx] if sim >= threshold else "UNKNOWN"
    return pred, float(sim)


def predict_centroid(test_emb, centroids, threshold):
    best_class, best_sim = None, -1
    for c, cent in centroids.items():
        sim = float(np.dot(cent, test_emb))
        if sim > best_sim:
            best_sim, best_class = sim, c
    pred = best_class if best_sim >= threshold else "UNKNOWN"
    return pred, best_sim


softmax_preds = {}

for img_name in test_names:
    crop, _  = get_scute_crop(f"{DATASET3_DIR}/{img_name}")
    tensor   = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier.model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    max_conf = float(np.max(probs))
    pred_idx = int(np.argmax(probs))

    SOFTMAX_THRESH = 0.70
    softmax_preds[img_name] = {
        "softmax_pred": CLASSES[pred_idx] if max_conf >= SOFTMAX_THRESH else "UNKNOWN",
        "softmax_conf": round(max_conf, 4)
    }
