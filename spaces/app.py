import numpy as np
import gradio as gr
from PIL import Image
import pathlib, json, urllib.request, gzip, struct

# ── Load TFLite model ─────────────────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter('multihead_myntra.tflite')
except Exception:
    import tensorflow as tf
    interp = tf.lite.Interpreter('multihead_myntra.tflite')

interp.allocate_tensors()
inp_det = interp.get_input_details()
out_det = interp.get_output_details()

# Identify which output is category (22 classes) and colour (21 classes)
out_by_size = {d['shape'][1]: d['index'] for d in out_det}
CAT_IDX = out_by_size[22]
COL_IDX = out_by_size[21]

# ── Load metadata ─────────────────────────────────────────────────────────────
meta         = json.load(open('myntra_meta.json'))
CLASS_NAMES  = meta['class_names']   # 22 subCategories
COLOUR_NAMES = meta['colour_names']  # 21 colours
COARSE_MAP   = meta['coarse_map']
IMG_SIZE     = meta['img_size']      # 96

CONFIDENCE_THRESHOLD = 0.50

COLOUR_SWATCHES = {
    'Black': '⬛', 'White': '⬜', 'Navy Blue': '🟦', 'Blue': '🔵',
    'Red': '🔴', 'Green': '🟢', 'Grey': '🩶', 'Brown': '🟫',
    'Pink': '🩷', 'Yellow': '🟡', 'Orange': '🟠', 'Purple': '🟣',
}

# ── Download Fashion-MNIST examples as fallback ───────────────────────────────
EXAMPLES_DIR = pathlib.Path('examples')
EXAMPLES_DIR.mkdir(exist_ok=True)
example_paths = []

try:
    BASE = 'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/'
    def _load_images(url, path):
        urllib.request.urlretrieve(url, path)
        with gzip.open(path, 'rb') as f:
            f.read(4); n = struct.unpack('>I', f.read(4))[0]
            r = struct.unpack('>I', f.read(4))[0]
            c = struct.unpack('>I', f.read(4))[0]
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
    def _load_labels(url, path):
        urllib.request.urlretrieve(url, path)
        with gzip.open(path, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    FMNIST_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    x_test = _load_images(BASE+'t10k-images-idx3-ubyte.gz', '/tmp/fmnist_img.gz')
    y_test = _load_labels(BASE+'t10k-labels-idx1-ubyte.gz', '/tmp/fmnist_lbl.gz')

    for cls in range(10):
        idx  = np.where(y_test == cls)[0][0]
        path = str(EXAMPLES_DIR / f"{FMNIST_NAMES[cls]}.png")
        Image.fromarray(x_test[idx], mode='L').resize((96, 96), Image.NEAREST).convert('RGB').save(path)
        example_paths.append(path)
except Exception as e:
    print(f'Could not load examples: {e}')

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image):
    img = Image.fromarray(image).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Upload a clothing image to see predictions."

    x = preprocess(image)
    interp.set_tensor(inp_det[0]['index'], x)
    interp.invoke()

    cat_probs = interp.get_tensor(CAT_IDX)[0]
    col_probs = interp.get_tensor(COL_IDX)[0]

    # Category
    cat_top3    = cat_probs.argsort()[::-1][:3]
    best_cat    = int(cat_top3[0])
    best_cat_c  = float(cat_probs[best_cat])
    cat_name    = CLASS_NAMES[best_cat]

    # Colour
    best_col    = int(col_probs.argmax())
    best_col_c  = float(col_probs[best_col])
    col_name    = COLOUR_NAMES[best_col]
    col_swatch  = COLOUR_SWATCHES.get(col_name, '🎨')

    lines = ["### Category\n"]
    for i in cat_top3:
        filled = int(cat_probs[i] * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        lines.append(f"**{CLASS_NAMES[i]}** — {cat_probs[i]:.1%}  `{bar}`")

    lines.append("\n### Colour\n")
    col_top3 = col_probs.argsort()[::-1][:3]
    for i in col_top3:
        filled = int(col_probs[i] * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        swatch = COLOUR_SWATCHES.get(COLOUR_NAMES[i], '🎨')
        lines.append(f"**{swatch} {COLOUR_NAMES[i]}** — {col_probs[i]:.1%}  `{bar}`")

    lines.append("\n---")
    if best_cat_c >= CONFIDENCE_THRESHOLD:
        lines.append(f"✅ **{cat_name}** · {col_swatch} **{col_name}** ({best_cat_c:.1%} confidence)")
    else:
        coarse = COARSE_MAP.get(cat_name, 'Apparel')
        lines.append(
            f"⚠️ Low confidence ({best_cat_c:.1%})\n\n"
            f"Coarse category: **{coarse}** · {col_swatch} **{col_name}**\n"
            f"_(best guess: {cat_name})_"
        )

    return "\n\n".join(lines)

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Myntra Clothing Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 👗 Myntra Fashion Classifier
    Trained on **44,000 real Myntra product photos** — predicts category AND colour in one pass.

    **Model:** MobileNetV2 multi-head · 96×96 RGB · 22 subCategories · 21 colours
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Clothing Image", type="numpy", height=300)
            if example_paths:
                gr.Examples(examples=example_paths, inputs=img_input, label="Sample images (click to try)")
        with gr.Column():
            output = gr.Markdown(value="*Upload or click a sample image.*")

    img_input.change(fn=predict, inputs=img_input, outputs=output)

    gr.Markdown("""
    ---
    [GitHub](https://github.com/v1neet3/ecommerce-clothing-classifier) · Built with MobileNetV2, TFLite & Gradio
    """)

demo.launch()
