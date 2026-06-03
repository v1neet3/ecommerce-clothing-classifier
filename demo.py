import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import pathlib

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model('simple_cnn.keras')

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
HIERARCHY = {
    'Tops':     ['T-shirt/top', 'Pullover', 'Coat', 'Shirt'],
    'Bottoms':  ['Trouser'],
    'Dresses':  ['Dress'],
    'Footwear': ['Sandal', 'Sneaker', 'Ankle boot'],
    'Bags':     ['Bag'],
}
CLASS_TO_COARSE = {c: k for k, v in HIERARCHY.items() for c in v}
EMOJI = {
    'T-shirt/top': '👕', 'Trouser': '👖', 'Pullover': '🧥',
    'Dress': '👗', 'Coat': '🧣', 'Sandal': '👡',
    'Shirt': '👔', 'Sneaker': '👟', 'Bag': '👜', 'Ankle boot': '👢',
}
CONFIDENCE_THRESHOLD = 0.60

# ── Save example images to disk ───────────────────────────────────────────────
EXAMPLES_DIR = pathlib.Path('examples')
EXAMPLES_DIR.mkdir(exist_ok=True)

print("Saving example images...")
(_, _), (x_test, y_test) = fashion_mnist.load_data()
example_paths = []
for cls in range(10):
    idx  = np.where(y_test == cls)[0][0]
    img  = x_test[idx]
    path = str(EXAMPLES_DIR / f"{CLASS_NAMES[cls].replace('/', '-')}.png")
    Image.fromarray(img, mode='L').resize((112, 112), Image.NEAREST).save(path)
    example_paths.append([path])

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image):
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [28, 28]) / 255.0
    return tf.expand_dims(img, 0).numpy()

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Upload a clothing image to see predictions."

    probs    = model.predict(preprocess(image), verbose=0)[0]
    top3     = probs.argsort()[::-1][:3]
    best_idx = int(top3[0])
    best_conf= float(probs[best_idx])
    best_name= CLASS_NAMES[best_idx]

    lines = ["### Predictions\n"]
    for i in top3:
        filled = int(probs[i] * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        lines.append(f"**{EMOJI.get(CLASS_NAMES[i],'')} {CLASS_NAMES[i]}** — {probs[i]:.1%}  `{bar}`")

    lines.append("\n---")
    if best_conf >= CONFIDENCE_THRESHOLD:
        lines.append(f"✅ **{EMOJI.get(best_name,'')} {best_name}** ({best_conf:.1%} confidence)")
    else:
        coarse = CLASS_TO_COARSE[best_name]
        lines.append(
            f"⚠️ Low confidence ({best_conf:.1%})\n\n"
            f"Showing coarse category: **{coarse}**  _(best guess: {EMOJI.get(best_name,'')} {best_name})_"
        )
    return "\n\n".join(lines)

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Clothing Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 👗 E-Commerce Clothing Classifier
    Upload any clothing image — classifies into **10 categories** with confidence scores.
    Falls back to a coarser label when confidence is below 60%.
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Clothing Image", type="numpy", height=300)
            gr.Examples(
                examples=example_paths,
                inputs=img_input,
                label="Click a sample to try",
            )
        with gr.Column():
            output = gr.Markdown(value="*Upload or click a sample image.*")

    img_input.change(fn=predict, inputs=img_input, outputs=output)

    gr.Markdown("""
    ---
    **Model:** Simple CNN · ~225K params · Fashion-MNIST · ~91% test accuracy
    """)

print("\n✅ Demo ready — open http://127.0.0.1:7860 in your browser\n")
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
