# 🚀 Instructions to Run Crocodile ReID Pipeline

## 📌 0. Setup

### Clone the repository

```bash
git clone "https://github.com/siyaa79/CV_2026_02_Visionformers"
cd crocodile-reid
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 1. Prepare Folder Structure

Create required folders (if not already present):

```bash
mkdir data weights outputs
```

---

## 📥 2. Add Dataset

Place your data as follows:

```text
data/
├── raw/           # original annotated images + labels (.jpg + .txt)
├── reid/          # images for cropping (input to YOLO)
├── test_raw/      # test images
```

---

## 🔀 3. Split Dataset (YOLO format)

```bash
python split_data.py
```

This creates:

```text
data/
├── images/train
├── images/val
├── labels/train
├── labels/val
```

---

## 🧠 4. Train YOLO Model (Scute Detection)

```bash
python train_yolo.py
```

After training:

* Copy best weights to:

```text
weights/best.pt
```

---

## ✂️ 5. Crop Scutes (Training Data for ReID)

```bash
python crop_scutes.py
```

Output:

```text
data/clean_crops/
```

---

## 🧬 6. Train ReID Model (ResNet50 + Triplet Loss)

```bash
python training_resnet50.py
```

Output:

```text
weights/reid_resnet50.pth
```

---

## 🧪 7. Crop Test Images

```bash
python test_data_crop.py
```

Output:

```text
data/test_crops/
```

---

## 🔍 8. Run ReID Testing

```bash
python reid_test.py
```

Output:

```text
outputs/reid_results.csv
```

---

## ⚠️ Notes

* Make sure all paths in code are **relative** (no `C:\Users\...`)
* GPU is recommended but not required
* Large datasets and weights are **not included in repo**

---

## 🛠️ Troubleshooting

### Module not found

```bash
pip install -r requirements.txt
```

### File not found errors

* Ensure correct folder structure
* Run commands from root directory:

```bash
cd crocodile-reid
```

---

## ✅ Full Pipeline (Quick Run)

Run everything step-by-step:

```bash
python split_data.py
python train_yolo.py
python crop_scutes.py
python training_resnet50.py
python test_data_crop.py
python reid_test.py
```

---

## 📌 Expected Outputs Summary

| Step          | Output                      |
| ------------- | --------------------------- |
| YOLO Training | `weights/best.pt`           |
| Cropping      | `data/clean_crops/`         |
| ReID Training | `weights/reid_resnet50.pth` |
| Testing       | `outputs/reid_results.csv`  |

---

## 🎯 Done

You now have a complete pipeline for:

* Detection (YOLO)
* Feature Learning (ResNet50)
* Re-identification (embedding matching)
