# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Azure ML online-endpoint scoring script for the car-part defect classifier.

Defines the two functions the managed online endpoint calls:
    init()          loads the TorchScript model + labels once at container start
    run(raw_data)   scores a single base64-encoded image per request

Expected request body (JSON), any of:
    {"image_base64": "<base64>"}
    {"input_data": {"data": ["<base64>"]}}     # AutoML-style envelope
    {"data": "<base64>"}
"""

import base64
import io
import json
import logging
import os

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


def init():
    """Load and cache the model. Called once when the container starts."""
    global model, preprocess, classes

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    # The registered model folder contains model.pt + labels.json.
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


def _find_file(root: str, filename: str, required: bool = True) -> str | None:
    """Locate a file within the model directory tree (path can vary by registration)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    if required:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return None


def _decode_image(raw_data) -> Image.Image:
    """Decode the request payload into a PIL RGB image."""
    if isinstance(raw_data, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw_data)).convert("RGB")

    payload = json.loads(raw_data)
    if "image_base64" in payload:
        b64 = payload["image_base64"]
    elif "input_data" in payload:  # AutoML-style envelope
        b64 = payload["input_data"]["data"][0]
    elif "data" in payload:
        b64 = payload["data"]
    else:
        raise ValueError("No image field found in request (expected image_base64/input_data/data)")

    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def run(raw_data):
    """Score one request. Returns predicted class + probabilities."""
    image = _decode_image(raw_data)
    tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    return {
        "predicted_class": classes[top_idx],
        "confidence": float(probs[top_idx].item()),
        "probabilities": {classes[i]: float(probs[i].item()) for i in range(len(classes))},
    }
