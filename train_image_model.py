import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset


BASE_MODEL = "Falconsai/nsfw_image_detection"
DATASET_DIR = "./dataset/_manual_downloads"
OUTPUT_DIR = "./fine_tuned_nsfw_model"
LOG_DIR = "./training_logs"

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.2          # 20% of data for validation

# Image settings
MAX_IMAGE_SIZE = 1024
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}

# Folder names that contain SAFE images
# Everything else is treated as UNSAFE
SAFE_FOLDER_NAMES = {"neutral", "normal", "safe", "non-violence", "non_violence",
                     "nonviolence", "negative", "benign", "clean"}


class NSFWDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert("RGB")

            # Resize very large images to save memory
            w, h = image.size
            if max(w, h) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(w, h)
                image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        except Exception as e:
            print(f"  Skipping corrupted image: {image_path}  {e}")
            blank = Image.new("RGB", (224, 224), (0, 0, 0))
            inputs = self.processor(images=blank, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
            }



def load_all_images(base_dir):
    """
    Walk through all subfolders and classify images as safe/unsafe
    based on the folder name they are in.
    """
    unsafe_paths = []
    safe_paths = []

    if not os.path.isdir(base_dir):
        return safe_paths, unsafe_paths

    for root, dirs, files in os.walk(base_dir):
        # Determine if this folder is safe or unsafe
        folder_name = os.path.basename(root).lower().strip()
        is_safe_folder = folder_name in SAFE_FOLDER_NAMES

        # Skip label folders (YOLO .txt files)
        if folder_name == "labels":
            continue

        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(root, filename)
                if is_safe_folder:
                    safe_paths.append(filepath)
                else:
                    unsafe_paths.append(filepath)

    return safe_paths, unsafe_paths


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    print("=" * 60)
    print("  SafeVision  Fine-tune Image Model")
    print("=" * 60)

    # --- Check GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("No GPU found -- training on CPU (will be slow)")
    print()

    # --- Load images from all subfolders ---
    print(f" Scanning: {os.path.abspath(DATASET_DIR)}")
    safe_paths, unsafe_paths = load_all_images(DATASET_DIR)

    unsafe_labels = [1] * len(unsafe_paths)
    safe_labels = [0] * len(safe_paths)

    print(f" Unsafe images: {len(unsafe_paths)}")
    print(f" Safe images:   {len(safe_paths)}")

    if len(unsafe_paths) == 0:
        print(f"\n No unsafe images found in {os.path.abspath(DATASET_DIR)}")
        return

    if len(safe_paths) == 0:
        print(f"\n No safe images found!")
        print(f"   Need a folder named 'neutral', 'normal', or 'safe' with images")
        return

    # --- Combine ---
    all_paths = unsafe_paths + safe_paths
    all_labels = unsafe_labels + safe_labels
    print(f"\n Total: {len(all_paths)} images ({len(unsafe_paths)} unsafe, {len(safe_paths)} safe)")

    # --- Split train/val ---
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=VAL_SPLIT, random_state=42, stratify=all_labels
    )
    print(f" Train: {len(train_paths)} | Validation: {len(val_paths)}")

    # --- Load model & processor ---
    print(f"\n Loading base model: {BASE_MODEL}")
    processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "normal", 1: "nsfw"},
        label2id={"normal": 0, "nsfw": 1},
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print(" Model loaded.")

    # --- Create datasets ---
    train_dataset = NSFWDataset(train_paths, train_labels, processor)
    val_dataset = NSFWDataset(val_paths, val_labels, processor)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,   # Windows compatibility
        remove_unused_columns=False,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    print(f"\n{'=' * 60}")
    print(f"    Starting training  {EPOCHS} epochs")
    print(f"{'=' * 60}\n")

    trainer.train()

    # --- Final evaluation ---
    print(f"\n{'=' * 60}")
    print(f"   Final Evaluation")
    print(f"{'=' * 60}\n")

    results = trainer.evaluate()
    print(f"  Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"  Precision: {results['eval_precision']:.4f}")
    print(f"  Recall:    {results['eval_recall']:.4f}")
    print(f"  F1 Score:  {results['eval_f1']:.4f}")

    # --- Confusion matrix ---
    val_preds = trainer.predict(val_dataset)
    predictions = np.argmax(val_preds.predictions, axis=-1)
    cm = confusion_matrix(val_labels, predictions)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Unsafe")
    print(f"  Actual Normal  {cm[0][0]:>4}  {cm[0][1]:>4}")
    print(f"  Actual Unsafe  {cm[1][0]:>4}  {cm[1][1]:>4}")

    # --- Save ---
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n Model saved to: {final_path}")
    print(f"\n To use it, update try.py line 29:")
    print(f'   model=\"./fine_tuned_nsfw_model/final\"')


if __name__ == "__main__":
    main()
