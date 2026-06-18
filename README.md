<p align="center">
  <h1 align="center">🛡️ SafeVision</h1>
  <p align="center">
    <strong>Real-time NSFW content moderation — text &amp; images — right in your browser.</strong>
  </p>
  <p align="center">
    A Chrome extension backed by a local GPU-accelerated Flask server that detects and blurs toxic text and NSFW images on any webpage in real time.
  </p>
</p>

---

## ✨ Features

| Capability | Details |
|---|---|
| **Text Moderation** | Detects toxicity, severe toxicity, and obscenity using [Detoxify](https://github.com/unitaryai/detoxify) |
| **Image Moderation** | Classifies images as safe/NSFW using a fine-tuned [ViT](https://huggingface.co/Falconsai/nsfw_image_detection) model (~98% accuracy) |
| **Real-time Blurring** | Unsafe content is CSS-blurred instantly — text gets a soft blur, images get a heavy 20px blur |
| **Batch Image Inference** | Groups images into batches for a single GPU forward pass — dramatically faster on pages with many images |
| **Smart Scanning** | Prioritizes visible/viewport content; chat sites scan newest messages first |
| **Chat Site Support** | Optimized for WhatsApp Web, Instagram, and Messenger with chat-switch detection and automatic rescanning |
| **Image Search Support** | Handles Google Images and Bing Images — re-scans on scroll, click, and lazy-loaded `src` changes |
| **GPU Accelerated** | CUDA + FP16 half-precision on discrete GPUs; automatic CPU fallback |
| **Custom Model Training** | Includes a full fine-tuning pipeline to train your own ViT-based NSFW classifier |

---

## 🏗️ Architecture

```
┌─────────────────────────────┐       HTTP (localhost:5000)       ┌──────────────────────────┐
│   Chrome Extension          │ ◄──────────────────────────────►  │   Flask Backend Server   │
│                             │                                   │                          │
│  content.js                 │   POST /predict_text              │  Detoxify (text model)   │
│   ├─ Scans DOM for text     │   POST /predict_image             │  ViT (image classifier)  │
│   ├─ Scans DOM for images   │   POST /predict_images_batch      │                          │
│   ├─ Batches & sends to BG  │                                   │  CUDA / CPU auto-detect  │
│   └─ Applies CSS blur       │                                   └──────────────────────────┘
│                             │
│  background.js              │
│   ├─ Relays API calls       │
│   └─ Keep-alive alarm       │
└─────────────────────────────┘
```

---

## 📋 Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (recommended) — CPU works but is significantly slower
- **Google Chrome** (or any Chromium-based browser)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gameodevil2022-a11y/Safe-Vision.git
cd Safe-Vision
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Activate it:**

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch (with CUDA)

> ⚠️ **Do NOT** install torch from `requirements.txt` — it installs a CPU-only build. Install the CUDA version separately:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

*Adjust `cu124` to match your CUDA version (e.g. `cu118`, `cu121`).*

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```env
HUGGINFACE_KEY = "your_huggingface_api_key"
```

### 6. Start the Backend Server

**Option A — Using the batch script (Windows):**

```bash
start.bat
```

**Option B — Manual:**

```bash
set CUDA_MODULE_LOADING=LAZY
python try.py
```

The server starts on `http://127.0.0.1:5000`.

On first run, models will be downloaded from Hugging Face (may take a few minutes). Subsequent starts are fast thanks to caching and a GPU warmup routine.

### 7. Load the Chrome Extension

1. Open Chrome → navigate to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `SafeVisionExtension/` folder
5. The extension is now active on all pages

---

## 🔌 API Endpoints

The Flask server exposes three endpoints, all on `http://127.0.0.1:5000`:

### `POST /predict_text`

Classify text for toxicity.

**Request:**
```json
{ "text": "some text to check" }
```

**Response:**
```json
{ "status": "safe" }
// or
{ "status": "unsafe", "score": 0.92 }
```

**Thresholds:** Toxicity, severe toxicity, or obscenity > **0.5** → `unsafe`

---

### `POST /predict_image`

Classify a single image.

**Request (URL):**
```json
{ "url": "https://example.com/photo.jpg" }
```

**Request (Base64):**
```json
{ "imageData": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Response:**
```json
{ "status": "safe" }
// or
{ "status": "unsafe", "score": 0.95 }
```

**Threshold:** Label = `nsfw` AND score > **0.8** → `unsafe`

---

### `POST /predict_images_batch`

Classify multiple images in one GPU forward pass.

**Request:**
```json
{
  "images": [
    { "id": "img1", "imageData": "data:image/jpeg;base64,..." },
    { "id": "img2", "imageData": "data:image/jpeg;base64,..." }
  ]
}
```

**Response:**
```json
{
  "results": [
    { "id": "img1", "status": "safe", "score": 0.12 },
    { "id": "img2", "status": "unsafe", "score": 0.97 }
  ]
}
```

---

## 🧠 Model Training (Custom Fine-Tuning)

SafeVision includes a complete training pipeline to fine-tune the ViT image classifier on your own dataset.

### Dataset Structure

```
dataset/
└── _manual_downloads/
    ├── safe/          ← Safe images (folder named: neutral, normal, safe, clean, etc.)
    ├── violence/      ← Unsafe images (any folder NOT in the safe list)
    └── nsfw/          ← Unsafe images
```

**Folder naming convention:**
- Folders named `neutral`, `normal`, `safe`, `non-violence`, `negative`, `benign`, or `clean` → labeled as **safe** (class 0)
- All other folders → labeled as **unsafe / NSFW** (class 1)

### Run Training

```bash
python train_image_model.py
```

**Hyperparameters** (configurable at the top of the script):

| Parameter | Default | Description |
|---|---|---|
| `EPOCHS` | 10 | Number of training epochs |
| `BATCH_SIZE` | 8 | Images per batch |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `WARMUP_RATIO` | 0.1 | Warm-up proportion of total steps |
| `WEIGHT_DECAY` | 0.01 | L2 regularization |
| `VAL_SPLIT` | 0.2 | Validation set fraction |

**Outputs:**
- Checkpoints saved to `fine_tuned_nsfw_model/`
- Best model (by F1 score) saved to `fine_tuned_nsfw_model/final/`
- Training logs in `training_logs/`

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix are printed at the end of training.

After training, the server (`try.py`) automatically loads the fine-tuned model from `./fine_tuned_nsfw_model/final` if it exists, otherwise falls back to the Hugging Face base model.

---

## 📂 Project Structure

```
Safe-Vision/
├── try.py                        # Flask backend server (text + image APIs)
├── train_image_model.py          # ViT fine-tuning pipeline
├── requirements.txt              # Python dependencies
├── start.bat                     # Windows one-click server launcher
├── .env                          # HuggingFace API key (git-ignored)
├── .gitignore
│
├── SafeVisionExtension/          # Chrome Extension (Manifest V3)
│   ├── manifest.json             # Extension config & permissions
│   ├── background.js             # Service worker — relays API calls
│   └── content.js                # Content script — DOM scanning & blurring
│
├── dataset/                      # Training data (git-ignored)
│   ├── safe/
│   └── unsafe/
│
├── fine_tuned_nsfw_model/        # Trained model checkpoints (git-ignored)
│   └── final/                    # Best model (auto-loaded by server)
│
└── Images/                       # Test images (git-ignored)
```

---

## ⚙️ How the Extension Works

1. **Content Script** (`content.js`) injects into every page and:
   - Walks the DOM via `TreeWalker` to find text nodes
   - Queries `img` elements in the viewport (skips icons, placeholders, tiny images)
   - Skips UI labels, dates, and JSON via a smart skip-list
   - Batches text checks (5 elements at a time, 200ms between batches)
   - Batches images (300ms collection window → single GPU forward pass)

2. **Background Service Worker** (`background.js`):
   - Proxies fetch requests to the local Flask server
   - Keeps alive via a `chrome.alarms` heartbeat every 24 seconds

3. **MutationObserver** watches for DOM changes:
   - New chat messages → re-scan chat panel
   - Chat switch detected (header name change) → full chat rescan
   - `src` attribute changes on `<img>` → re-classify the image

4. **Blurring:**
   - Text: `filter: blur(5px)` + disabled selection and pointer events
   - Images: `filter: blur(20px)` + disabled pointer events

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML — Text** | [Detoxify](https://github.com/unitaryai/detoxify) (Unitary AI) |
| **ML — Image** | [ViT](https://huggingface.co/Falconsai/nsfw_image_detection) (Vision Transformer) via Hugging Face Transformers |
| **Training** | Hugging Face `Trainer` API with scikit-learn metrics |
| **Backend** | Flask + Flask-CORS |
| **GPU** | PyTorch with CUDA, FP16 half-precision, lazy kernel loading |
| **Extension** | Chrome Manifest V3 (content script + service worker) |

---

## 📝 License

This project is for educational and research purposes.

---

<p align="center">
  Built with ❤️ for a safer internet.
</p>
