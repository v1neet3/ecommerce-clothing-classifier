# 👗 E-Commerce Clothing Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-3.10-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-Demo-F97316?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-~91%25-22C55E?style=for-the-badge"/>
</p>

<p align="center">
  An end-to-end image classification system for fashion retail — from raw pixels to a production-ready API with drift monitoring.
</p>

---

## What It Does

Upload a clothing image → get an instant prediction with confidence score, attribute tags, and a smart fallback label when the model is uncertain.

```
Raw image  →  CNN / MobileNetV2  →  Category + Sleeve + Formality
                                  →  Confidence score
                                  →  Hierarchical fallback if unsure
                                  →  Drift alert if distribution shifts
```

---

## Models

| Model | Params | Test Accuracy | Best For |
|---|---|---|---|
| **Simple CNN** (scratch) | ~225K | ~91% | Edge / low-latency |
| **MobileNetV2** (transfer) | ~3.5M | ~88–91% | Server-side accuracy |

Both trained on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale 28×28 images across 10 clothing categories.

### Classes

| Label | Label | Label | Label | Label |
|---|---|---|---|---|
| 👕 T-shirt/top | 👖 Trouser | 🧥 Pullover | 👗 Dress | 🧣 Coat |
| 👡 Sandal | 👔 Shirt | 👟 Sneaker | 👜 Bag | 👢 Ankle boot |

---

## Features

### Core
- **Two model architectures** trained and compared side-by-side (accuracy, loss, params)
- **BatchNorm + Dropout** for stable training and regularisation
- **EarlyStopping** with best-weight restore

### Production Pipeline

| Feature | Description |
|---|---|
| 🔄 **Data Augmentation** | 6-way policy baked into model graph — flip, rotate, zoom, translate, brightness, contrast |
| 🏷️ **Multi-Label Heads** | Single backbone → 3 outputs: category + sleeve type + formality |
| 🌳 **Hierarchical Fallback** | Returns coarse label (e.g. "Tops") when confidence < 60% |
| 🚀 **TF Serving Export** | `SavedModel` format, ready for Docker/REST deployment |
| 📦 **ONNX Export** | Portable format — runs in PyTorch, browsers, mobile, edge devices |
| 📊 **Drift Monitoring** | Logs every prediction, tracks confidence & class distribution, fires alerts |

---

## Live Demo

```bash
pip install -r requirements.txt
python demo.py
```

Open **http://127.0.0.1:7860** — upload any clothing photo or click a sample image to classify it instantly.

---

## Run the Notebook

```bash
pip install -r requirements.txt
jupyter notebook ecommerce_clothing_classifier.ipynb
```

Or open `ecommerce_clothing_classifier_executed.ipynb` to browse all outputs, plots, confusion matrices, and training logs without running anything.

---

## Project Structure

```
├── ecommerce_clothing_classifier.ipynb       # Full notebook (source)
├── ecommerce_clothing_classifier_executed.ipynb  # Pre-run with all outputs
├── demo.py                                   # Gradio live demo
├── simple_cnn.keras                          # Trained Simple CNN (~1.2MB)
└── requirements.txt                          # Dependencies
```

---

## Results

### Training Curves
Accuracy and loss plotted for both models across epochs — with EarlyStopping.

### Confusion Matrix
The hardest classes to separate are **Shirt ↔ T-shirt/top ↔ Pullover** (visually similar in 28×28 grayscale). Footwear classes (Sandal, Sneaker, Ankle boot) score highest.

### Confidence & Fallback
A threshold sweep shows the trade-off between fine-label coverage and fallback rate — tunable for your precision/recall requirements.

---

## What's Next (for a real catalog)

- **Higher-res data** — real product photos at 256×256+ RGB (try [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))
- **Real attribute labels** — human-annotated sleeve/formality/color for the multi-label head
- **EfficientNetV2** — better accuracy at similar speed vs MobileNetV2
- **Quantisation** — 4× size reduction via TFLite for on-device inference
- **CLIP embeddings** — enables text-to-image search ("red summer dress")
- **Async retraining** — trigger retraining automatically when drift is detected

---

## Requirements

- Python 3.9+
- TensorFlow 2.20 / Keras 3.10
- See `requirements.txt` for the full list

---

<p align="center">Made with ❤️ using TensorFlow, Keras & Gradio</p>
