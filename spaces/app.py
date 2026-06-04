import numpy as np
import gradio as gr
from gradio import mount_gradio_app
from fastapi import FastAPI, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pathlib, json, io, os, urllib.request, gzip, struct

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
out_by_size = {d['shape'][1]: d['index'] for d in out_det}
CAT_IDX = out_by_size[22]
COL_IDX = out_by_size[21]

# ── Metadata ──────────────────────────────────────────────────────────────────
meta         = json.load(open('myntra_meta.json'))
CLASS_NAMES  = meta['class_names']
COLOUR_NAMES = meta['colour_names']
COARSE_MAP   = meta['coarse_map']
IMG_SIZE     = meta['img_size']

CONFIDENCE_THRESHOLD = 0.50
COLOUR_SWATCHES = {
    'Black': '⬛', 'White': '⬜', 'Navy Blue': '🟦', 'Blue': '🔵',
    'Red': '🔴', 'Green': '🟢', 'Grey': '🩶', 'Brown': '🟫',
    'Pink': '🩷', 'Yellow': '🟡', 'Orange': '🟠', 'Purple': '🟣',
}

# API key — set via HF Spaces secret "API_KEY", falls back to demo key
API_KEY = os.environ.get('API_KEY', 'demo-key-2024')

# ── Core inference (shared by UI and API) ─────────────────────────────────────
def run_inference(pil_image: Image.Image) -> dict:
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    x   = np.array(img, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0

    interp.set_tensor(inp_det[0]['index'], x)
    interp.invoke()
    cat_probs = interp.get_tensor(CAT_IDX)[0]
    col_probs = interp.get_tensor(COL_IDX)[0]

    cat_top3 = cat_probs.argsort()[::-1][:3].tolist()
    col_top3 = col_probs.argsort()[::-1][:3].tolist()
    best_cat = cat_top3[0]
    best_col = col_top3[0]
    cat_conf = float(cat_probs[best_cat])
    cat_name = CLASS_NAMES[best_cat]

    return {
        'category':   cat_name,
        'colour':     COLOUR_NAMES[best_col],
        'confidence': round(cat_conf, 4),
        'fallback':   cat_conf < CONFIDENCE_THRESHOLD,
        'coarse_category': COARSE_MAP.get(cat_name, 'Apparel'),
        'top3_categories': [
            {'label': CLASS_NAMES[i], 'score': round(float(cat_probs[i]), 4)}
            for i in cat_top3
        ],
        'top3_colours': [
            {'label': COLOUR_NAMES[i], 'score': round(float(col_probs[i]), 4)}
            for i in col_top3
        ],
    }

# ── Example images ────────────────────────────────────────────────────────────
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
    print(f'Examples unavailable: {e}')

# ── Gradio predict (for UI) ───────────────────────────────────────────────────
def predict_ui(image):
    if image is None:
        return "Upload a clothing image to see predictions."

    result = run_inference(Image.fromarray(image))
    cat_name = result['category']
    col_name = result['colour']
    col_s    = COLOUR_SWATCHES.get(col_name, '🎨')

    lines = ["### Category\n"]
    for item in result['top3_categories']:
        filled = int(item['score'] * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        lines.append(f"**{item['label']}** — {item['score']:.1%}  `{bar}`")

    lines.append("\n### Colour\n")
    for item in result['top3_colours']:
        filled = int(item['score'] * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        sw     = COLOUR_SWATCHES.get(item['label'], '🎨')
        lines.append(f"**{sw} {item['label']}** — {item['score']:.1%}  `{bar}`")

    lines.append("\n---")
    if not result['fallback']:
        lines.append(f"✅ **{cat_name}** · {col_s} **{col_name}** ({result['confidence']:.1%} confidence)")
    else:
        lines.append(
            f"⚠️ Low confidence ({result['confidence']:.1%})\n\n"
            f"Coarse category: **{result['coarse_category']}** · {col_s} **{col_name}**\n"
            f"_(best guess: {cat_name})_"
        )
    return "\n\n".join(lines)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Myntra Clothing Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 👗 Myntra Fashion Classifier
    Trained on **44,000 real Myntra product photos** — predicts category AND colour in one pass.

    **Model:** MobileNetV2 multi-head · 96×96 RGB · 22 subCategories · 21 colours

    > **API available** — see the [GitHub README](https://github.com/v1neet3/ecommerce-clothing-classifier#api) for integration docs.
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Clothing Image", type="numpy", height=300)
            if example_paths:
                gr.Examples(examples=example_paths, inputs=img_input, label="Sample images")
        with gr.Column():
            output = gr.Markdown(value="*Upload or click a sample image.*")

    img_input.change(fn=predict_ui, inputs=img_input, outputs=output)

    gr.Markdown("""
    ---
    [GitHub](https://github.com/v1neet3/ecommerce-clothing-classifier) ·
    [API Docs](https://github.com/v1neet3/ecommerce-clothing-classifier#api) ·
    Built with MobileNetV2, TFLite & Gradio
    """)

# ── FastAPI app with Gradio mounted ───────────────────────────────────────────
fastapi_app = FastAPI(
    title="Fashion Classifier API",
    description="Classify clothing images into category and colour using a MobileNetV2 model trained on Myntra product photos.",
    version="1.0.0",
)

@fastapi_app.get("/api/health")
def health():
    return {"status": "ok", "model": "MobileNetV2-MultiHead-Myntra", "version": "1.0.0"}

@fastapi_app.post("/api/predict")
async def predict_api(
    file: UploadFile,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """
    Classify a clothing image.

    - **file**: image file (JPEG/PNG)
    - **X-API-Key**: your API key (header)

    Returns category, colour, confidence, top-3 predictions, and fallback label.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass your key in the X-API-Key header.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG).")

    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG or PNG.")

    result = run_inference(img)
    return JSONResponse(result)

# Mount Gradio UI at root — API routes stay at /api/*
app = mount_gradio_app(fastapi_app, demo, path="/")
