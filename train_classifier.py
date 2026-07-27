#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Transfer-learning trainer for the car-parts / car-assembly quality dataset.

Trains a 3-class defect-severity classifier (no_defect / minor_defect /
major_defect) on the ImageFolder-structured dataset produced by
``car_assembly_dataset_generator.py``. Uses an ImageNet-pretrained backbone whose
last few blocks are fine-tuned (with discriminative learning rates) on top of a
fresh classification head, plus label smoothing and test-time augmentation to
sharpen the minor-vs-no-defect boundary on a small dataset.

Outputs (written to ``azureml/model/`` by default, ready to register on Azure ML):
    - model.pt      TorchScript model (no Python class needed at inference time)
    - labels.json   ordered class names matching the model's output logits

Usage:
    python train_classifier.py
    python train_classifier.py --backbone mobilenet_v3_large --epochs 40
    python train_classifier.py --data-dir car_parts_dataset --output-dir azureml/model
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONES = ("efficientnet_b0", "mobilenet_v3_large", "resnet50")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Train a car-part defect classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("car_parts_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("azureml/model"))
    parser.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Train/eval resolution; 256 matches the dataset's native size "
                             "(no upscaling), giving the model full detail for subtle defects.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head-lr", type=float, default=1e-3, help="LR for the fresh classifier head.")
    parser.add_argument("--backbone-lr", type=float, default=1e-4,
                        help="Low LR for unfrozen backbone blocks (discriminative fine-tuning).")
    parser.add_argument("--finetune-blocks", type=int, default=2,
                        help="Unfreeze the last N backbone blocks for fine-tuning. 0 keeps the "
                             "backbone fully frozen (feature-extraction only).")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Softens targets; helps confusable adjacent classes.")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear LR warmup epochs before cosine decay.")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                        help="Test-time augmentation: average predictions over a horizontal flip.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=8, help="Early-stop patience on val macro-F1.")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Build training (augmented) and evaluation transforms."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # TrivialAugmentWide: a strong, tuning-free auto-augmentation policy that samples
        # a diverse mix of geometric/color operations each step. On small datasets it
        # exposes the model to far more variation, sharpening the subtle minor-vs-clean
        # boundary without hand-tuning individual augment magnitudes.
        transforms.TrivialAugmentWide(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_model(name: str, n_out: int, dropout: float,
                finetune_blocks: int) -> tuple[nn.Module, nn.Module]:
    """Build a pretrained backbone with a fresh head.

    The head is always trainable. When ``finetune_blocks`` > 0 the last N feature
    blocks of the backbone are unfrozen for discriminative fine-tuning; the
    earlier (general-purpose) layers stay frozen. Returns ``(model, head)`` so the
    caller can assign a higher learning rate to the head than to the backbone.
    """
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_out))
        head = m.classifier
        blocks = list(m.features)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, n_out)
        head = m.classifier
        blocks = list(m.features)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_out))
        head = m.fc
        blocks = [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        raise ValueError(f"Unknown backbone: {name}")

    # Freeze the whole backbone, then selectively re-enable gradients.
    for p in m.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    if finetune_blocks > 0:
        for block in blocks[-finetune_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    return m, head


def run_epoch(model, loader, criterion, optimizer, device, *, train: bool) -> tuple[float, float]:
    """Run one epoch; returns (mean_loss, macro_f1)."""
    model.train(train)
    total_loss, preds, gts = 0.0, [], []
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


def export_for_serving(model, class_names, img_size, output_dir: Path) -> None:
    """Export a TorchScript model + labels.json ready for Azure ML registration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")
    example = torch.rand(1, 3, img_size, img_size)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
    scripted = torch.jit.freeze(scripted)
    model_path = output_dir / "model.pt"
    scripted.save(str(model_path))
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"   \u2713 {model_path}")
    print(f"   \u2713 {labels_path}")


def run(args: argparse.Namespace) -> int:
    """Train, evaluate, and export the classifier."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Training {args.backbone} on {args.data_dir} (device: {device})")
    print("=" * 60)

    if not args.data_dir.exists():
        print(f"Error: dataset directory not found: {args.data_dir}", file=sys.stderr)
        return EXIT_ERROR

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(args.data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=eval_tf)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Images: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}\n")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, head = build_model(args.backbone, num_classes, args.dropout, args.finetune_blocks)
    model = model.to(device)

    # Class-weighted, label-smoothed loss guards against mild imbalance and curbs
    # overconfident errors between look-alike classes (minor vs no defect).
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=num_classes).float()
    class_weights = (counts.sum() / (num_classes * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Discriminative learning rates: nudge the pretrained backbone gently, train
    # the fresh head faster.
    head_ids = {id(p) for p in head.parameters()}
    head_params = [p for p in head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    param_groups = [{"params": head_params, "lr": args.head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
        print(f"Fine-tuning last {args.finetune_blocks} backbone block(s): "
              f"{len(backbone_params)} tensors at lr {args.backbone_lr:g}")
    else:
        print("Backbone frozen (feature-extraction only).")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Linear warmup then cosine decay, applied as a multiplier on each group's base LR.
    def lr_scale(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    best_f1, best_state, epochs_no_improve = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_ld, criterion, optimizer, device, train=True)
        va_loss, va_f1 = run_epoch(model, val_ld, criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"epoch {epoch:02d}  train_loss {tr_loss:.3f} f1 {tr_f1:.3f} | "
              f"val_loss {va_loss:.3f} f1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1, best_state, epochs_no_improve = va_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (best val macro-F1 {best_f1:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds, test_gts = [], []
    with torch.no_grad():
        for x, y in test_ld:
            x = x.to(device)
            probs = model(x).softmax(1)
            if args.tta:
                # Average with a horizontal-flip view for a small, free accuracy gain.
                probs = (probs + model(torch.flip(x, dims=[3])).softmax(1)) / 2
            test_preds += probs.argmax(1).cpu().tolist()
            test_gts += y.tolist()

    print(f"\nTest classification report{' (TTA)' if args.tta else ''}:")
    print(classification_report(test_gts, test_preds, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(test_gts, test_preds))

    print("\nExporting model for serving...")
    export_for_serving(model, class_names, args.img_size, args.output_dir)
    print(f"\nDone. Best val macro-F1: {best_f1:.3f}")
    return EXIT_SUCCESS


def main() -> int:
    """Main entry point."""
    args = create_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
