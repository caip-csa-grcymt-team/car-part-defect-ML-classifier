#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Deploy the car-part defect classifier to an Azure ML BATCH endpoint.

Batch endpoints score a whole folder of images on an existing compute cluster
that scales to zero when idle -- so they need NO online-endpoint quota. This is
the deployment path to use when managed online-endpoint CPU quota is unavailable.

End-to-end SDK v2 flow:
    1. Connect to the workspace
    2. Resolve the model (registered name latest, or register from --model-dir)
    3. Register / reuse the inference environment (conda.yaml)
    4. Create a batch endpoint
    5. Create a batch deployment bound to an existing compute cluster
    6. Route default traffic to the deployment
    7. (Optional) Submit a scoring job over a local folder of images

Prerequisites:
    pip install azure-ai-ml azure-identity
    az login

Usage:
    python azureml/deploy_batch_endpoint.py \
        --subscription-id <sub> --resource-group <rg> --workspace <ws> \
        --registered-model car-parts-pytorch

    # deploy and immediately score a folder of images:
    python azureml/deploy_batch_endpoint.py --subscription-id <sub> \
        --resource-group <rg> --workspace <ws> \
        --registered-model car-parts-pytorch --test-folder car_parts_dataset/test
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

HERE = Path(__file__).resolve().parent


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Deploy defect classifier to an Azure ML batch endpoint.")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint-name", default="carparts-batch",
                        help="Batch endpoint name (default: carparts-batch).")
    parser.add_argument("--deployment-name", default="blue")
    parser.add_argument("--model-dir", type=Path, default=HERE / "model")
    parser.add_argument("--registered-model", default=None,
                        help="Use an already-registered model by name (latest version) "
                             "instead of uploading --model-dir.")
    parser.add_argument("--score-dir", type=Path, default=HERE / "batchscoring")
    parser.add_argument("--conda-file", type=Path, default=HERE / "conda.yaml")
    parser.add_argument("--compute", default="cpu-cluster",
                        help="Existing compute cluster to run batch jobs on.")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=10)
    parser.add_argument("--test-folder", type=Path, default=None,
                        help="Optional local folder of images to score after deploy.")
    return parser


def run(args: argparse.Namespace) -> int:
    """Register model + environment and deploy the batch endpoint."""
    from azure.ai.ml import Input, MLClient
    from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
    from azure.ai.ml.entities import (
        BatchDeployment,
        BatchEndpoint,
        BatchRetrySettings,
        CodeConfiguration,
        Environment,
        Model,
    )
    from azure.identity import DefaultAzureCredential

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    if args.registered_model:
        print(f"Using registered model '{args.registered_model}' (latest version)...")
        model = ml_client.models.get(name=args.registered_model, label="latest")
    else:
        model_path = args.model_dir / "model.pt"
        if not model_path.exists():
            print(f"Error: {model_path} not found. Train first, or pass --registered-model.", file=sys.stderr)
            return EXIT_ERROR
        print("Registering model from local files...")
        model = ml_client.models.create_or_update(
            Model(path=str(args.model_dir), type=AssetTypes.CUSTOM_MODEL, name="car-parts-pytorch")
        )
    print(f"   \u2713 {model.name}:{model.version}")

    print("Registering inference environment...")
    env = ml_client.environments.create_or_update(
        Environment(
            name="pytorch-batch-env",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
            conda_file=str(args.conda_file),
        )
    )
    print(f"   \u2713 {env.name}:{env.version}")

    print(f"Creating batch endpoint '{args.endpoint_name}'...")
    ml_client.batch_endpoints.begin_create_or_update(
        BatchEndpoint(name=args.endpoint_name, description="Car-part defect classifier (batch)")
    ).result()

    print(f"Creating batch deployment '{args.deployment_name}' on compute '{args.compute}'...")
    deployment = BatchDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=model.id,
        environment=env.id,
        code_configuration=CodeConfiguration(code=str(args.score_dir), scoring_script="batch_score.py"),
        compute=args.compute,
        instance_count=args.instance_count,
        max_concurrency_per_instance=2,
        mini_batch_size=args.mini_batch_size,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
        logging_level="info",
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()

    print("Setting default deployment...")
    endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
    endpoint.defaults.deployment_name = args.deployment_name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
    print("\n" + "=" * 60)
    print("Batch endpoint deployed.")
    print(f"   Endpoint   : {args.endpoint_name}")
    print(f"   Scoring URI: {endpoint.scoring_uri}")
    print(f"   Compute    : {args.compute} (scales to zero when idle)")
    print("=" * 60)

    if args.test_folder:
        if not args.test_folder.exists():
            print(f"\nWarning: test folder not found: {args.test_folder}", file=sys.stderr)
        else:
            _submit_job(ml_client, args.endpoint_name, args.test_folder, Input, AssetTypes)

    return EXIT_SUCCESS


def _submit_job(ml_client, endpoint_name: str, folder: Path, Input, AssetTypes) -> None:
    """Submit a batch scoring job over a local folder of images."""
    print(f"\nSubmitting scoring job over '{folder}'...")
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=endpoint_name,
        input=Input(type=AssetTypes.URI_FOLDER, path=str(folder)),
    )
    print(f"   Job submitted: {job.name}")
    print("   Track it in Azure ML Studio (Endpoints -> Batch -> Jobs), or run:")
    print(f"     ml_client.jobs.stream('{job.name}')")
    print("   When done, download outputs (predictions.csv) from the job.")


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
