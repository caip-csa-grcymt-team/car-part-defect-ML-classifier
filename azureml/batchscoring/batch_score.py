# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Azure ML batch-endpoint scoring script for the car-part defect classifier.

Batch endpoints call:
    init()            loads the TorchScript model + labels once per worker
    run(mini_batch)   scores a list of local image file paths, returns rows

Unlike the online scoring script, batch scoring receives file PATHS (the batch
runtime downloads the input folder to local disk) rather than base64 payloads.
Each returned DataFrame row is appended to the output predictions file.
"""

import json
import logging
import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

model = None
preprocess = None
classes = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CLASSES = ["major_defect", "minor_defect", "no_defect"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def init():
    """Load and cache the model. Called once per worker process."""
    global model, preprocess, classes

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = _find_file(model_dir, "model.pt")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    labels_path = _find_file(model_dir, "labels.json", required=False)
    if labels_path:
        with open(labels_path, encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = DEFAULT_CLASSES

    preprocess = transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    logging.info("Init complete. Classes: %s", classes)


def _find_file(root: str, filename: str, required: bool = True):
    """Locate a file within the model directory tree (path can vary by registration)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    if required:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return None


def run(mini_batch):
    """Score a batch of image files. Returns a DataFrame of predictions."""
    rows = []
    for file_path in mini_batch:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in VALID_EXT:
            continue
        try:
            image = Image.open(file_path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            row = {
                "file": os.path.basename(file_path),
                "predicted_class": classes[top_idx],
                "confidence": round(float(probs[top_idx].item()), 4),
            }
            for i, cls in enumerate(classes):
                row[f"p_{cls}"] = round(float(probs[i].item()), 4)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", file_path, exc)
            rows.append({"file": os.path.basename(file_path), "predicted_class": "ERROR",
                         "confidence": 0.0})
    return pd.DataFrame(rows)
