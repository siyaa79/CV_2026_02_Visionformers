# =========================
# 1. IMPORTS
# =========================
import os
import random
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. DATASET (FIXED LABELS)
# =========================
class CrocDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        self.label_map = {}
        current_label = 0

        for label in os.listdir(root):
            label_path = os.path.join(root, label)

            if not os.path.isdir(label_path):
                continue

            # 🔥 Map 107 → 0, 108 → 1, etc.
            if label not in self.label_map:
                self.label_map[label] = current_label
                current_label += 1

            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                self.samples.append((img_path, self.label_map[label]))

        print("Label mapping:", self.label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================
# 3. BALANCED SAMPLER
# =========================
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.label_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.labels_set = list(self.label_to_indices.keys())
        self.n_classes = n_classes
        self.n_samples = n_samples

    def __iter__(self):
        while True:
            classes = random.sample(self.labels_set, self.n_classes)
            batch = []

            for cls in classes:
                indices = random.sample(self.label_to_indices[cls], self.n_samples)
                batch.extend(indices)

            yield batch

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_samples)


# =========================
# 4. MODEL (ResNet50)
# =========================
class ReIDModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x, dim=1)
        return x


# =========================
# 5. TRIPLET CREATION
# =========================
def create_triplets(embeddings, labels, margin=1.0):
    triplets = []
    labels = labels.cpu()

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    for i in range(len(embeddings)):
        anchor = embeddings[i]
        label = labels[i]

        pos_mask = (labels == label)
        neg_mask = (labels != label)

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        # 🔥 Hard positive
        pos_dists = dist_matrix[i][pos_indices]
        hard_pos_idx = pos_indices[pos_dists.argmax()]
        pos = embeddings[hard_pos_idx]

        # 🔥 Semi-hard negative
        neg_dists = dist_matrix[i][neg_indices]

        ap_dist = dist_matrix[i][hard_pos_idx]

        # condition: negative farther than positive BUT within margin
        semi_hard_mask = (neg_dists > ap_dist) & (neg_dists < ap_dist + margin)

        if semi_hard_mask.any():
            semi_neg_idx = neg_indices[semi_hard_mask.nonzero(as_tuple=True)[0][0]]
        else:
            # fallback: random negative
            semi_neg_idx = neg_indices[neg_dists.argmin()]

        neg = embeddings[semi_neg_idx]

        triplets.append((anchor, pos, neg))

    return triplets

# =========================
# 6. TRANSFORMS (NO DISTORTION)
# =========================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


# =========================
# 7. LOAD DATA (UPDATED PATH)
# =========================
dataset_path = "./data/clean_crops"

dataset = CrocDataset(dataset_path, transform)

labels = [label for _, label in dataset.samples]

sampler = BalancedBatchSampler(labels, n_classes=4, n_samples=3)

loader = DataLoader(dataset, batch_sampler=sampler)


# =========================
# 8. MODEL + LOSS
# =========================
model = ReIDModel().to(device)

criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# =========================
# 9. TRAINING LOOP
# =========================
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        embeddings = model(imgs)

        triplets = create_triplets(embeddings, labels)

        if len(triplets) == 0:
            continue

        loss = 0
        for a, p, n in triplets:
            loss += criterion(a.unsqueeze(0), p.unsqueeze(0), n.unsqueeze(0))

        loss /= len(triplets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

        # prevent infinite loop
        if batch_idx > 100:
            break

    print(f"Epoch {epoch} DONE | Avg Loss: {total_loss:.4f}")


# =========================
# 10. SAVE MODEL
# =========================
save_path = "./weights/reid_resnet50.pth"

torch.save(model.state_dict(), save_path)

print("✅ Training complete. Model saved!")