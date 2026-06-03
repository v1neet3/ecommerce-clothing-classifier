import numpy as np
import gradio as gr
from PIL import Image
import pathlib, urllib.request, gzip, struct

# ── Load weights (pure numpy — no TF needed) ──────────────────────────────────
W = dict(np.load('weights.npz'))

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

# ── Numpy CNN forward pass ─────────────────────────────────────────────────────
def _relu(x):    return np.maximum(0, x)
def _maxpool(x): return x.reshape(x.shape[0], x.shape[1]//2, 2, x.shape[2]//2, 2, x.shape[3]).max(axis=(2,4))
def _gap(x):     return x.mean(axis=(1, 2))
def _bn(x, g, b, m, v, eps=1e-3): return g * (x - m) / np.sqrt(v + eps) + b
def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _conv2d(x, kernel, bias):
    N, H, W, _ = x.shape
    kH, kW, _, _ = kernel.shape
    xp  = np.pad(x, ((0,0),(1,1),(1,1),(0,0)))
    out = sum(np.einsum('nhwi,io->nhwo', xp[:,r:r+H,c:c+W,:], kernel[r,c,:,:])
              for r in range(kH) for c in range(kW))
    return out + bias

def predict_probs(x):
    x = _bn(_relu(_conv2d(x, W['conv1_kernel'], W['conv1_bias'])),
            W['bn1_gamma'], W['bn1_beta'], W['bn1_mean'], W['bn1_var'])
    x = _maxpool(x)
    x = _bn(_relu(_conv2d(x, W['conv2_kernel'], W['conv2_bias'])),
            W['bn2_gamma'], W['bn2_beta'], W['bn2_mean'], W['bn2_var'])
    x = _maxpool(x)
    x = _relu(_conv2d(x, W['conv3_kernel'], W['conv3_bias']))
    x = _gap(x)
    x = _relu(x @ W['dense_kernel'] + W['dense_bias'])
    return _softmax(x @ W['out_kernel'] + W['out_bias'])

# ── Example images (Fashion-MNIST via direct download) ────────────────────────
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

    x_test = _load_images(BASE+'t10k-images-idx3-ubyte.gz', '/tmp/fmnist_img.gz')
    y_test = _load_labels(BASE+'t10k-labels-idx1-ubyte.gz', '/tmp/fmnist_lbl.gz')

    for cls in range(10):
        idx  = np.where(y_test == cls)[0][0]
        path = str(EXAMPLES_DIR / f"{CLASS_NAMES[cls].replace('/', '-')}.png")
        Image.fromarray(x_test[idx], mode='L').resize((112, 112), Image.NEAREST).save(path)
        example_paths.append(path)
except Exception as e:
    print(f'Examples unavailable: {e}')

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image):
    img = Image.fromarray(image).convert('L').resize((28, 28), Image.LANCZOS)
    return np.array(img, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Upload a clothing image to see predictions."

    probs    = predict_probs(preprocess(image))[0]
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

    **Model:** Simple CNN · ~225K params · Fashion-MNIST · ~91% test accuracy
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Clothing Image", type="numpy", height=300)
            if example_paths:
                gr.Examples(examples=example_paths, inputs=img_input, label="Click a sample to try")
        with gr.Column():
            output = gr.Markdown(value="*Upload or click a sample image.*")

    img_input.change(fn=predict, inputs=img_input, outputs=output)

    gr.Markdown("""
    ---
    [GitHub](https://github.com/v1neet3/ecommerce-clothing-classifier) · Built with NumPy & Gradio
    """)

demo.launch()
