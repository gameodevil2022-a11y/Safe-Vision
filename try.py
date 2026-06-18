import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"   # Faster startup — load GPU kernels on demand

from flask import Flask, request, jsonify
from flask_cors import CORS
from detoxify import Detoxify
from transformers import pipeline
from dotenv import load_dotenv
from PIL import Image
import requests
import torch
import time
from io import BytesIO

load_dotenv()
KEY1 = os.getenv("HUGGINFACE_KEY")

# --- GPU DETECTION (GPU first, CPU as fallback) ---
INTEGRATED_GPU_KEYWORDS = ["intel", "uhd", "iris", "vega", "integrated"]

def get_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if any(k in gpu_name for k in INTEGRATED_GPU_KEYWORDS):
            print(f"Integrated GPU ({torch.cuda.get_device_name(0)}) — using CPU")
            return "cpu"
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM) — CUDA enabled")
        return "cuda"
    print("No discrete GPU found — using CPU")
    return "cpu"

DEVICE = get_device()
USE_FP16 = (DEVICE == "cuda")  # Half precision on GPU only

# --- LOAD MODELS ---
print("Loading models...")

# 1. Text Model
try:
    text_model = Detoxify("original", device=DEVICE)
    print("TEXT Model loaded.")
except Exception as e:
    print(f"Text model failed: {e}")

# 2. Image Model — fine-tuned SafeVision model (98% accuracy)
FINE_TUNED_MODEL = "./fine_tuned_nsfw_model/final"
FALLBACK_MODEL   = "Falconsai/nsfw_image_detection"

try:
    model_path = FINE_TUNED_MODEL if os.path.isdir(FINE_TUNED_MODEL) else FALLBACK_MODEL
    device_arg = 0 if DEVICE == "cuda" else -1  # pipeline uses int device index

    image_classifier = pipeline(
        "image-classification",
        model=model_path,
        device=device_arg,
        dtype=torch.float16 if USE_FP16 else torch.float32,
    )

    # Warmup — eliminate first-request cold start (GPU JIT compile, etc.)
    print("Warming up model...")
    try:
        dummy = Image.new("RGB", (224, 224), (128, 128, 128))
        for _ in range(3):
            image_classifier(dummy)
        print(f"Image Model ready: {model_path} | device={DEVICE} | fp16={USE_FP16}")
    except Exception as warmup_err:
        print(f"Warmup skipped ({warmup_err}) — model will warm up on first request")
        print(f"Image Model loaded: {model_path} | device={DEVICE}")

except Exception as e:
    print(f"Image model failed: {e}")

# --- SERVER SETUP ---
app = Flask(__name__)
CORS(app)


@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text: return jsonify({'status': 'safe'}), 200

        t_start = time.time()
        results = text_model.predict(text)
        t_end = time.time()
        
        # Log for debugging
        tox = results['toxicity']
        sev = results['severe_toxicity']
        obs = results['obscene']
        latency_ms = (t_end - t_start) * 1000
        print(f"📝 TEXT: \"{text[:80]}\" → tox={tox:.3f}, sev={sev:.3f}, obs={obs:.3f} [{latency_ms:.1f}ms]")
        
        # Threshold for text (0.5 = 50% confidence, lowered for better recall)
        if tox > 0.5 or sev > 0.5 or obs > 0.5:
             print(f"   🚫 UNSAFE!")
             return jsonify({'status': 'unsafe', 'score': float(tox)})
        
        print(f"   ✅ safe")
        return jsonify({'status': 'safe'})
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return jsonify({'error': str(e)}), 500

import base64 as _base64

def decode_image(image_data):
    """Decode a base64 data URL into a PIL Image."""
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]
    img_bytes = _base64.b64decode(image_data)
    return Image.open(BytesIO(img_bytes)).convert('RGB')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        image_url = data.get('url', '')
        image_data = data.get('imageData', '')

        img = None
        if image_data:
            img = decode_image(image_data)
            print(f"IMAGE: [base64, {len(image_data)} chars]")
        elif image_url:
            resp = requests.get(image_url, timeout=5)
            if resp.status_code != 200:
                return jsonify({'status': 'safe'}), 200
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            print(f"IMAGE: {image_url[:80]}")
        else:
            return jsonify({'status': 'safe'}), 200

        result = image_classifier(img)[0]
        print(f"  -> label={result['label']}, score={result['score']:.3f}")
        if result['label'] == 'nsfw' and result['score'] > 0.8:
            return jsonify({'status': 'unsafe', 'score': float(result['score'])})
        return jsonify({'status': 'safe'})

    except Exception as e:
        print(f"Image error: {e}")
        return jsonify({'status': 'safe', 'error': str(e)}), 200


@app.route('/predict_images_batch', methods=['POST'])
def predict_images_batch():
    """
    Accepts: { "images": [ {"id": "...", "imageData": "data:..."}, ... ] }
    Returns: { "results": [ {"id": "...", "status": "safe/unsafe"}, ... ] }
    """
    try:
        data = request.get_json()
        items = data.get('images', [])
        if not items:
            return jsonify({'results': []}), 200

        t_total_start = time.time()

        # Decode all images
        t_decode_start = time.time()
        pil_images = []
        ids = []
        for item in items:
            try:
                img = decode_image(item['imageData'])
                pil_images.append(img)
                ids.append(item['id'])
            except Exception as e:
                print(f"Batch decode error for id={item.get('id')}: {e}")
                ids.append(item['id'])
                pil_images.append(None)
        t_decode_end = time.time()

        # Run batch inference — GPU processes all at once
        results_out = []
        valid_imgs = [(i, img) for i, img in enumerate(pil_images) if img is not None]

        t_infer_start = time.time()
        if valid_imgs:
            indices, imgs = zip(*valid_imgs)
            batch_results = image_classifier(list(imgs))  # single GPU forward pass

            result_map = {}
            for idx, res in zip(indices, batch_results):
                top = res[0] if isinstance(res, list) else res
                result_map[idx] = top
        else:
            result_map = {}
        t_infer_end = time.time()

        for i, img_id in enumerate(ids):
            if pil_images[i] is None:
                results_out.append({'id': img_id, 'status': 'safe'})
                continue
            top = result_map.get(i, {'label': 'normal', 'score': 1.0})
            label = top['label']
            score = top['score']
            status = 'unsafe' if label == 'nsfw' and score > 0.8 else 'safe'
            print(f"  [{img_id[:8]}] {label} {score:.3f} -> {status}")
            results_out.append({'id': img_id, 'status': status, 'score': float(score)})

        t_total_end = time.time()
        n = len(valid_imgs)
        decode_ms = (t_decode_end - t_decode_start) * 1000
        infer_ms = (t_infer_end - t_infer_start) * 1000
        total_ms = (t_total_end - t_total_start) * 1000
        per_img = infer_ms / n if n > 0 else 0
        print(f"  ⏱️ BATCH {n} imgs | decode={decode_ms:.0f}ms | infer={infer_ms:.0f}ms | total={total_ms:.0f}ms | per_img={per_img:.1f}ms")

        return jsonify({'results': results_out})

    except Exception as e:
        print(f"Batch error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    app.run(host='0.0.0.0', port=5000, debug=False)