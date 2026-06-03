# E-Commerce Clothing Classifier

End-to-end image classification system for fashion retail, built on Fashion-MNIST.

## Models

| Model | Params | Test Accuracy |
|---|---|---|
| Simple CNN (from scratch) | ~225K | ~91% |
| MobileNetV2 Transfer Learning | ~3.5M | ~88–91% |

## Features

- **Data augmentation** baked into the model graph (flip, rotate, zoom, brightness, contrast)
- **Multi-label heads** — category + sleeve type + formality from one backbone
- **Hierarchical fallback** — coarse label (e.g. "Tops") when confidence < 60%
- **TF Serving export** — SavedModel ready for Docker deployment
- **Drift monitoring** — logs every prediction, alerts on accuracy drop or distribution shift

## Live Demo

```bash
pip install -r requirements.txt
python demo.py
# Open http://127.0.0.1:7860
```

## Run the Notebook

```bash
jupyter notebook ecommerce_clothing_classifier.ipynb
```

Or view the pre-executed version: `ecommerce_clothing_classifier_executed.ipynb`

## Project Structure

```
├── ecommerce_clothing_classifier.ipynb      # Main notebook (source)
├── ecommerce_clothing_classifier_executed.ipynb  # Pre-run with all outputs
├── demo.py                                  # Gradio live demo
├── simple_cnn.keras                         # Trained Simple CNN
└── requirements.txt                         # Dependencies
```

## Requirements

- Python 3.9+
- TensorFlow 2.20
- See `requirements.txt` for full list

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale 28×28 images across 10 clothing categories. Built into Keras, no download needed.
