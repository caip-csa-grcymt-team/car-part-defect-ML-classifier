#!/usr/bin/env python3

"""Score a folder of car-part images against a running LOCAL  endpoint.

This is the "get predictions" step: point it at the test dataset (or any folder
of images) and it asks the local endpoint for a defect-type prediction on each
image, then prints predicted vs. actual and an overall accuracy summary.

The folder is expected to be organised by class, e.g.:

    car_parts_dataset/test/
        major_defect/*.jpg
        minor_defect/*.jpg
        no_defect/*.jpg

The subfolder name is treated as the ground-truth label. Images placed directly
in the top-level folder (no class subfolder) are scored with actual = "unknown".

Prerequisites:
    A local endpoint already deployed via deploy_local_endpoint.py
    pip install azure-ai-ml azure-identity
    Docker Desktop running
    az login

Usage:
    python azureml/score_test_folder.py \
        --subscription-id <sub> --resource-group <rg> --workspace <ws> \
        --images-dir car_parts_dataset/test
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Score an image folder against a local Azure ML endpoint.")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint-name", default="carparts-local")
    parser.add_argument("--deployment-name", default="blue")
    parser.add_argument("--images-dir", type=Path, default=Path("car_parts_dataset/test"),
                        help="Folder of images, ideally organised into per-class subfolders.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally score only the first N images (quick smoke test).")
    parser.add_argument("--output-csv", type=Path, default=Path("test_predictions.csv"),
                        help="Where to write the per-image predictions.")
    return parser


def collect_images(images_dir: Path) -> list[tuple[Path, str]]:
    """Return (image_path, actual_label) pairs. Label is the parent folder name."""
    pairs: list[tuple[Path, str]] = []
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            actual = path.parent.name if path.parent != images_dir else "unknown"
            pairs.append((path, actual))
    return pairs


def score_image(ml_client, endpoint_name: str, deployment_name: str, image_path: Path) -> dict:
    """Invoke the local endpoint with one image and return the parsed response."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"image_base64": b64}, f)
        request_file = f.name
    raw = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=request_file,
        local=True,
    )
    return raw if isinstance(raw, dict) else json.loads(raw)


def run(args: argparse.Namespace) -> int:
    """Score every image in the folder and report predicted vs. actual."""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    if not args.images_dir.is_dir():
        print(f"Error: images dir not found: {args.images_dir}", file=sys.stderr)
        return EXIT_ERROR

    pairs = collect_images(args.images_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"Error: no images found under {args.images_dir}", file=sys.stderr)
        return EXIT_ERROR

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    print(f"Scoring {len(pairs)} image(s) against local endpoint '{args.endpoint_name}'...\n")
    rows: list[dict] = []
    correct = 0
    scored_with_labels = 0

    for image_path, actual in pairs:
        result = score_image(ml_client, args.endpoint_name, args.deployment_name, image_path)
        predicted = result.get("predicted_class", "?")
        confidence = result.get("confidence", 0.0)

        is_known = actual != "unknown"
        if is_known:
            scored_with_labels += 1
            if predicted == actual:
                correct += 1
            mark = "OK " if predicted == actual else "XX "
        else:
            mark = "-- "

        print(f"{mark}{image_path.name:40s} predicted={predicted:12s} actual={actual:12s} conf={confidence:.2f}")
        rows.append({
            "image": str(image_path),
            "actual": actual,
            "predicted": predicted,
            "confidence": f"{confidence:.4f}",
        })

    _write_csv(args.output_csv, rows)

    print("\n" + "=" * 60)
    if scored_with_labels:
        accuracy = correct / scored_with_labels
        print(f"Accuracy: {correct}/{scored_with_labels} = {accuracy:.1%} (on labelled images)")
    else:
        print("No labelled images (all 'unknown') - predictions written without accuracy.")
    print(f"Predictions written to: {args.output_csv}")
    print("=" * 60)
    return EXIT_SUCCESS


def _write_csv(output_csv: Path, rows: list[dict]) -> None:
    """Write per-image predictions to a CSV file."""
    import csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "actual", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Main entry point."""
    args = create_parser().parse_args()
    try:
        return run(args)
    except ImportError:
        print("Error: install the SDK first -> pip install azure-ai-ml azure-identity", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
