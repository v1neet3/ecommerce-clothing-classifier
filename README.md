# 👗 E-Commerce Clothing Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-3.10-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <a href="https://huggingface.co/spaces/v1neet3/ecommerce-clothing-classifier"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-F97316?style=for-the-badge"/></a>
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

## Datasets & Models

### Fashion-MNIST (baseline)
| Model | Params | Test Accuracy | Best For |
|---|---|---|---|
| **Simple CNN** (scratch) | ~225K | ~91% | Edge / low-latency |
| **MobileNetV2** (transfer) | ~3.5M | ~88–91% | Server-side accuracy |

Trained on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale 28×28 images across 10 clothing categories.

### Myntra Fashion Products (upgrade)
44,000 real RGB product photos · 96×96 · 20+ subCategories + colour labels

| Model | Notes |
|---|---|
| **MobileNetV2** | Fast, mobile-friendly · head + fine-tune |
| **EfficientNetV2-S** | Higher capacity · head + fine-tune |
| **Multi-head MobileNetV2** | Category + Colour predicted from one backbone |

> Download dataset: `kaggle datasets download -d paramaggarwal/fashion-product-images-small`

### Classes (Fashion-MNIST)

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

🚀 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/v1neet3/ecommerce-clothing-classifier)** — no setup needed, runs in your browser.

Or run locally:

```bash
pip install -r requirements.txt
python demo.py
# Open http://127.0.0.1:7860
```

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
├── ecommerce_clothing_classifier.ipynb           # Fashion-MNIST notebook (source)
├── ecommerce_clothing_classifier_executed.ipynb  # Pre-run with all outputs
├── myntra_clothing_classifier.ipynb              # Myntra dataset notebook (source)
├── myntra_clothing_classifier_executed.ipynb     # Pre-run with all outputs
├── myntra_meta.json                              # Class names, colour map, metadata
├── demo.py                                       # Gradio local demo
├── simple_cnn.keras                              # Trained Simple CNN (~1.2MB)
├── mobilenet_myntra.keras                        # MobileNetV2 on Myntra (22MB)
├── multihead_myntra.keras                        # Multi-head model — category + colour (15MB)
├── requirements.txt                              # Dependencies
└── spaces/
    ├── app.py                                    # Gradio UI + FastAPI REST endpoint
    ├── multihead_myntra.tflite                   # Quantised MobileNetV2 (2.9MB)
    └── requirements.txt                          # HF Spaces dependencies
```

---

## API

A REST API runs alongside the live demo. Integrate it into any app with a single HTTP call.

**Base URL**
```
https://v1neet3-ecommerce-clothing-classifier.hf.space
```

**Authentication**
Pass your API key in the `X-API-Key` header. Request a key by opening an issue on this repo.

For testing, use the demo key: `demo-key-2024`

---

### `POST /api/predict`

Classify a clothing image. Returns category, colour, confidence, and top-3 predictions.

**Request**
```
POST /api/predict
X-API-Key: demo-key-2024
Content-Type: multipart/form-data

file: <image file>
```

**Response**
```json
{
  "category": "Topwear",
  "colour": "Navy Blue",
  "confidence": 0.87,
  "fallback": false,
  "coarse_category": "Apparel",
  "top3_categories": [
    {"label": "Topwear",   "score": 0.87},
    {"label": "Bottomwear","score": 0.08},
    {"label": "Innerwear", "score": 0.03}
  ],
  "top3_colours": [
    {"label": "Navy Blue", "score": 0.91},
    {"label": "Blue",      "score": 0.06},
    {"label": "Black",     "score": 0.02}
  ]
}
```

> When `fallback: true`, confidence was below 50% and `coarse_category` is returned as the safe label.

---

### `GET /api/health`

Check if the API is up.

```json
{"status": "ok", "model": "MobileNetV2-MultiHead-Myntra", "version": "1.0.0"}
```

---

### Code Examples

**cURL**
```bash
curl -X POST \
  https://v1neet3-ecommerce-clothing-classifier.hf.space/api/predict \
  -H "X-API-Key: demo-key-2024" \
  -F "file=@jacket.jpg"
```

**Python**
```python
import requests

url  = "https://v1neet3-ecommerce-clothing-classifier.hf.space/api/predict"
headers = {"X-API-Key": "demo-key-2024"}

with open("jacket.jpg", "rb") as f:
    response = requests.post(url, headers=headers, files={"file": f})

result = response.json()
print(f"{result['category']} · {result['colour']} ({result['confidence']:.0%})")
```

**JavaScript (fetch)**
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch(
  "https://v1neet3-ecommerce-clothing-classifier.hf.space/api/predict",
  {
    method: "POST",
    headers: { "X-API-Key": "demo-key-2024" },
    body: formData,
  }
);
const result = await response.json();
console.log(`${result.category} · ${result.colour}`);
```

**React Native**
```javascript
const formData = new FormData();
formData.append("file", { uri: imageUri, type: "image/jpeg", name: "photo.jpg" });

const res = await fetch(
  "https://v1neet3-ecommerce-clothing-classifier.hf.space/api/predict",
  { method: "POST", headers: { "X-API-Key": "demo-key-2024" }, body: formData }
);
const { category, colour, confidence } = await res.json();
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
