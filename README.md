<img width="933" height="331" alt="image" src="https://github.com/user-attachments/assets/85d70747-672e-48db-8529-65bff0dff112" /># People Detection & Counting — YOLOv8 vs YOLOv5 vs RT-DETR

This repository presents a complete workflow for **real-time person detection, tracking, and counting** using three modern object detection architectures — **YOLOv8**, **YOLOv5**, and **RT-DETR**.  
The project includes **data preparation**, **exploratory data analysis (EDA)**, **training and evaluation**, and a fully interactive **Streamlit deployment** for real-world testing.

---

## Project Overview

- **Objective**: Build a robust, real-time people detection system and benchmark different state-of-the-art architectures on the same dataset.
- **Models Explored**:
  - [YOLOv8](https://github.com/ultralytics/ultralytics)
  - [YOLOv5](https://github.com/ultralytics/yolov5)
  - [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **Deployment**: Interactive [Streamlit](https://streamlit.io/) web app for:
  - Image detection
  - Video detection & live frame-by-frame counting
  - People counting using line crossing with **DeepSORT** tracking

## Why We Built This Project & Practical Applications

Accurately detecting and counting people in images and videos is a **core computer vision challenge with direct real-world impact**. Traditional manual counting is inefficient and error-prone, while commercial solutions are often expensive or closed-source.  
This project aims to build a **transparent, open, and reproducible pipeline** that allows anyone to train and deploy state-of-the-art person detection models tailored to their needs.
We trained all three models — **YOLOv5, YOLOv8, and RT-DETR** — on a **custom person-only COCO dataset** using **Kaggle Notebooks with NVIDIA Tesla T4 GPUs**, making the training reproducible and cost-efficient.
By comparing three leading architectures — **YOLOv5, YOLOv8, and RT-DETR** — on a controlled, **person-focused dataset**, we not only understand their trade-offs but also create a ready-to-use system for:

- **Crowd analytics & safety** — monitor occupancy in public venues, stadiums, airports, and events to ensure safe capacity.
- **Retail & smart spaces** — count foot traffic, analyze customer flow, and optimize store layouts or staffing.
- **Security & surveillance** — detect people in restricted zones, trigger alerts, and enhance camera monitoring systems.
- **Transportation & urban planning** — measure pedestrian movement across crosswalks, metro entrances, and other public areas.
- **Research & benchmarking** — provide a reproducible, flexible baseline for further experimentation in object detection.

Our approach shows how to go from **raw COCO data → filtered person-only dataset → training YOLO/transformer models → real-time application deployment**.

---

## Dataset & Preprocessing

We used the **COCO 2017 dataset** but **filtered it to only the `person` class** to make the system task-focused.

- Original: 5000 images, 36,781 annotations, 80 categories.
- Processed: Person-only images with regenerated YOLO format labels.

### Key Steps

- **Data download & filtering** (keeping only `person` class)
- **Conversion to YOLO label format**
- **Train/validation split (80/20)**
- **EDA on COCO (person distribution, annotation stats)**

**EDA Visuals:**

- EDA — Category Frequency

  <img width="1313" height="768" alt="category_frequency" src="https://github.com/user-attachments/assets/87a547f4-df6a-45cd-9477-002e55e9f639" />

- EDA — Sample annotated images

  <img width="1938" height="790" alt="images_from_dataset" src="https://github.com/user-attachments/assets/4d2183d6-14fe-4a1c-8438-2086d9bad641" />

---

## 3. Model Training & Evaluation

We trained **three models** under controlled settings for fair comparison.

---

### YOLOv8

- Framework: `ultralytics`
- Config: `yolov8-person.yaml`
- Epochs: 100
- Image size: 640×640
- Optimizer: SGD

**Training Visualizations:**

- Training Loss Curves

  <img width="691" height="470" alt="yolov8_training_loss" src="https://github.com/user-attachments/assets/8ff78621-f62a-4726-bdc4-7bb36f1674fb" />

- Validation Metrics (Precision, Recall, mAP)

  <img width="691" height="470" alt="validation_metrics_yolov8" src="https://github.com/user-attachments/assets/ef4bd64d-7b96-4228-9867-4c1b92053898" />

- Precision-Recall Curve

   <img width="691" height="470" alt="precision_recall_yolov8" src="https://github.com/user-attachments/assets/ffc6a152-227e-4f22-bf03-96447656b79b" />

- Confusion Matrix

  <img width="636" height="504" alt="confusion_matrix_yolov8" src="https://github.com/user-attachments/assets/00895b39-9ffb-41a5-85fd-2ec13c36fe4a" />

**Final Metrics (YOLOv8):**

```
Precision:           0.7333
Recall:              0.6202
mAP@50:              0.6795
mAP@50-95:           0.4401
```

---

### YOLOv5

- Framework: `ultralytics/yolov5`
- Config: `person_v5.yaml`
- Epochs: 100
- Image size: 640×640

**Training Visualizations:**

- Training Loss Curves

  <img width="700" height="470" alt="training_loss_yolov5" src="https://github.com/user-attachments/assets/c30b5c1f-1b95-4fcc-ab1a-4123e5385c17" />

- Validation Metrics

  <img width="691" height="470" alt="validation_metrics_yolov5" src="https://github.com/user-attachments/assets/6e8f2a78-61b7-4954-b734-d8d8c108b8ad" />

**Final Metrics (YOLOv5):**

```
Precision:           0.801
Recall:              0.637
mAP@50:              0.715
mAP@50-95:           0.484
```

---

### RT-DETR

- Framework: `ultralytics` (Real-Time DEtection TRansformer)
- Model: RT-DETR-Lite
- Epochs: 25 (fast convergence)

**Training Visualizations:**

- Training Loss Curves

  <img width="704" height="470" alt="rtdetr_training_loss" src="https://github.com/user-attachments/assets/a502dc5b-30b6-4eff-a1af-03802a3f9d09" />

- Validation Metrics

  <img width="695" height="470" alt="rtdetr_validation_metrics" src="https://github.com/user-attachments/assets/c01813c8-5a03-45c7-992f-24781322d714" />

- Precision-Recall Curve

  <img width="695" height="470" alt="rtdetr_precision_curves" src="https://github.com/user-attachments/assets/461f176f-0ec7-4e02-87e1-9ba5ee6496ab" />

- Confusion Matrix

  <img width="636" height="504" alt="rtdetr_confusion" src="https://github.com/user-attachments/assets/c2a5ec97-cdfc-47de-8bd0-ec5f4bb79177" />

**Final Metrics (RT-DETR):**

```
Precision:           0.6621
Recall:              0.5220
mAP@50:              0.5858
mAP@50-95:           0.3494
```

---

## Model Performance Comparison & Analysis

We benchmarked three state-of-the-art detectors — **YOLOv5**, **YOLOv8**, and **RT-DETR** — trained on the same **person-only subset** of COCO.

### Final Validation Metrics

| Model   | Precision | Recall    | mAP@50    | mAP@50–95 |
| ------- | --------- | --------- | --------- | --------- |
| YOLOv8  | 0.7333    | 0.6202    | 0.6795    | 0.4401    |
| YOLOv5  | **0.801** | **0.637** | **0.715** | **0.484** |
| RT-DETR | 0.6621    | 0.5220    | 0.5858    | 0.3494    |

---

### Interpretation & Takeaways

#### YOLOv5 — Mature and Highly Optimized

- Achieved the **best overall performance** across all metrics.
- High **precision (0.801)** means fewer false positives.
- Likely benefits from years of community-driven optimization.

#### YOLOv8 — Competitive but More Lightweight

- Only slightly behind YOLOv5 (precision down ~7%, mAP50-95 down ~9%).
- Easier to train, cleaner API, faster to deploy.

#### RT-DETR — Promising but Needs Longer Training

- Transformer-based, better at long-range context.
- Lower performance due to shorter training (25 epochs).
- Scales well with more training data & tuning.

---

### ⚖️ Practical Impact

- **Highest detection accuracy & stability** → **YOLOv5**
- **Modern tooling, faster deployment** → **YOLOv8**
- **Future-proof transformer approach** → **RT-DETR** (after longer training)

---

## 4. Deployment — Streamlit App

We developed a fully interactive **Streamlit application** for testing these models.

### Features

- Upload an **image** and detect people.
- Upload a **video** for frame-wise counting.
- Upload a **video with a virtual line** for entry/exit counting (DeepSORT tracking).

**UI Screenshots (add your images here):**

- App main interface

  <img width="1512" height="949" alt="Screenshot 2025-10-04 at 12 42 40 PM" src="https://github.com/user-attachments/assets/f23d93fc-0664-4968-bc31-daf9d6266939" />

- Image detection example

  <img width="1512" height="949" alt="Screenshot 2025-10-04 at 12 43 05 PM" src="https://github.com/user-attachments/assets/33947fba-6ba8-459e-a996-0aa46d6aa2a7" />

- Video detection example

  <img width="1512" height="949" alt="Screenshot 2025-10-04 at 12 43 56 PM" src="https://github.com/user-attachments/assets/92ff96fe-9ea9-46fa-bec7-3c2f042009e7" />

- Line-cross counting example
  <img width="1512" height="949" alt="Screenshot 2025-10-04 at 7 50 12 PM" src="https://github.com/user-attachments/assets/29c71f81-0c89-4273-8926-e8c02e1174b6" />

**Run the app:**

```bash
streamlit run app.py
```

Switch between models by changing:

```python
MODEL_PATH = "yolov8_person_best.pt"  # or yolov5_person_best.pt / rtdetr_person_best.pt
```

---

## 5. Installation

```bash
# Install dependencies
pip install ultralytics
pip install streamlit opencv-python-headless deep-sort-realtime
```

---

## 6. Highlights & Contributions

- Performed **EDA on COCO** to extract & analyze the `person` class distribution.
- Designed **clean training pipelines** for YOLOv5, YOLOv8, and RT-DETR.
- Conducted **systematic performance benchmarking** (Precision, Recall, mAP).
- Developed a **real-time people counting app** with image/video/line-crossing support.
- Produced **visualizations** for model metrics, confusion matrices, and predictions.

---

## 👨‍💻 Authors

- **[Ojas Godase](https://github.com/ojas-godase)** — _Author_

  - Email: ojas.godase13@gmail.com
  - LinkedIn: [Linkedin](https://www.linkedin.com/in/ojas01010/)

- **[Rudra Shekhare](https://github.com/RudraShekhare)** — _Co-Author_
  - Email: rudra282805@gmail.com
  - LinkedIn: [Linkedin](https://www.linkedin.com/in/rudra-shekhare-006987279/)

## Citation

If you use this work, please cite it as:

> Ojas Godase and Rudra Shekhare , **"Real-Time Person Detection and Counting Using YOLOv5, YOLOv8, and RT-DETR"**, Zenodo, 2025.  
> DOI: [10.5281/zenodo.17266539](https://doi.org/10.5281/zenodo.17266539)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17266539.svg)](https://doi.org/10.5281/zenodo.17266539)
