#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Submit the car-part defect classifier training as an Azure ML command job.

End-to-end SDK v2 flow (training happens IN Azure ML, not locally):
    1. Connect to the workspace
    2. Ensure a CPU compute cluster exists (scale-to-zero)
    3. Register the dataset folder as a uri_folder data asset
    4. Register the training environment (train_env.yaml)
    5. Submit a command job that runs train_classifier.py on the cluster
    6. (Optional) Stream the job to completion and register the trained model

Prerequisites:
    pip install azure-ai-ml azure-identity
    az login

Usage:
    # submit and return immediately (monitor in Azure ML Studio):
    python azureml/submit_training_job.py \
        --subscription-id <sub> --resource-group <rg> --workspace <ws>

    # submit, wait for completion, then register the model:
    python azureml/submit_training_job.py --subscription-id <sub> \
        --resource-group <rg> --workspace <ws> --wait --register
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Submit defect-classifier training to Azure ML.")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "car_parts_dataset",
                        help="Local dataset folder to upload as a data asset.")
    parser.add_argument("--data-asset", default=None,
                        help="Reuse an already-registered data asset by 'name' (latest) or "
                             "'name:version' instead of uploading --data-dir. Avoids re-upload "
                             "when the workspace storage has key-based auth disabled.")
    parser.add_argument("--compute-name", default="cpu-cluster")
    parser.add_argument("--compute-size", default="Standard_DS3_v2", help="CPU SKU; sufficient for a small CNN.")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=["efficientnet_b0", "mobilenet_v3_large", "resnet50"])
    parser.add_argument("--img-size", type=int, default=256,
                        help="Train/eval resolution; 256 matches the dataset's native image size.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--finetune-blocks", type=int, default=2,
                        help="Unfreeze the last N backbone blocks for fine-tuning (0 = frozen).")
    parser.add_argument("--backbone-lr", type=float, default=1e-4,
                        help="Low LR for the unfrozen backbone blocks.")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for the training loss.")
    parser.add_argument("--setup-only", action="store_true",
                        help="Provision compute + data asset + environment, then stop (no job submitted).")
    parser.add_argument("--wait", action="store_true", help="Stream job logs until it finishes.")
    parser.add_argument("--register", action="store_true",
                        help="Register the trained model after the job completes (implies --wait).")
    parser.add_argument("--model-name", default="car-parts-pytorch")
    return parser


def run(args: argparse.Namespace) -> int:
    """Register data + environment and submit the training command job."""
    from azure.ai.ml import MLClient, Input, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import AmlCompute, Data, Environment, Model
    from azure.identity import DefaultAzureCredential

    if not args.data_asset and not (args.data_dir / "train").exists():
        print(f"Error: {args.data_dir}/train not found. Point --data-dir at the dataset root.", file=sys.stderr)
        return EXIT_ERROR

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    print(f"Ensuring compute cluster '{args.compute_name}' exists...")
    try:
        ml_client.compute.get(args.compute_name)
        print("   \u2713 already exists")
    except Exception:
        cluster = AmlCompute(
            name=args.compute_name,
            type="amlcompute",
            size=args.compute_size,
            min_instances=0,
            max_instances=1,
            idle_time_before_scale_down=120,  # scale to zero after 2 min idle
        )
        ml_client.compute.begin_create_or_update(cluster).result()
        print(f"   \u2713 created {args.compute_name} ({args.compute_size}, scale-to-zero)")

    if args.data_asset:
        if ":" in args.data_asset:
            name, version = args.data_asset.split(":", 1)
            print(f"Using existing data asset '{name}:{version}'...")
            data_asset = ml_client.data.get(name=name, version=version)
        else:
            print(f"Using latest version of existing data asset '{args.data_asset}'...")
            data_asset = ml_client.data.get(name=args.data_asset, label="latest")
        print(f"   \u2713 {data_asset.name}:{data_asset.version}")
    else:
        print("Registering dataset as a uri_folder data asset...")
        data_asset = ml_client.data.create_or_update(
            Data(path=str(args.data_dir), type=AssetTypes.URI_FOLDER, name="car-parts-images",
                 description="Car-part quality images (train/val/test).")
        )
        print(f"   \u2713 {data_asset.name}:{data_asset.version}")

    print("Registering training environment...")
    env = ml_client.environments.create_or_update(
        Environment(
            name="pytorch-train-env",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
            conda_file=str(HERE / "train_env.yaml"),
        )
    )
    print(f"   \u2713 {env.name}:{env.version}")

    if args.setup_only:
        print("\n" + "=" * 60)
        print("Setup complete (no training job submitted).")
        print(f"   Compute     : {args.compute_name} ({args.compute_size}, scale-to-zero)")
        print(f"   Data asset  : {data_asset.name}:{data_asset.version}")
        print(f"   Environment : {env.name}:{env.version}")
        print("=" * 60)
        print("Review the above in Azure ML Studio, then re-run WITHOUT --setup-only to train.")
        return EXIT_SUCCESS

    print("Submitting training command job...")
    job = command(
        code=str(REPO_ROOT),  # uploads train_classifier.py (+ repo) as the job code
        command=(
            "python train_classifier.py "
            "--data-dir ${{inputs.data}} "
            "--output-dir ./outputs/model "
            f"--backbone {args.backbone} --epochs {args.epochs} "
            f"--img-size {args.img_size} "
            f"--finetune-blocks {args.finetune_blocks} --backbone-lr {args.backbone_lr:g} "
            f"--label-smoothing {args.label_smoothing:g}"
        ),
        inputs={"data": Input(type=AssetTypes.URI_FOLDER, path=data_asset.id)},
        environment=env.id,
        compute=args.compute_name,
        display_name="car-parts-defect-classifier",
        experiment_name="car-parts-training",
    )
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"   \u2713 submitted job: {returned_job.name}")
    print(f"   Studio: {returned_job.studio_url}")

    if not (args.wait or args.register):
        print("\nJob is running in Azure ML. Monitor it in the Studio link above.")
        return EXIT_SUCCESS

    print("\nStreaming job logs until completion (Ctrl+C stops streaming, not the job)...")
    ml_client.jobs.stream(returned_job.name)

    final = ml_client.jobs.get(returned_job.name)
    print(f"\nJob finished with status: {final.status}")
    if final.status != "Completed":
        print("Job did not complete successfully; skipping model registration.", file=sys.stderr)
        return EXIT_FAILURE

    if args.register:
        print("Registering trained model...")
        model = ml_client.models.create_or_update(
            Model(
                path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/outputs/model/",
                type=AssetTypes.CUSTOM_MODEL,
                name=args.model_name,
            )
        )
        print(f"   \u2713 registered {model.name}:{model.version}")
        print("\nNext: deploy with azureml/deploy_endpoint.py "
              "(it will pick up the latest registered model).")

    return EXIT_SUCCESS


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
