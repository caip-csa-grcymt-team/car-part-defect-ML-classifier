#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Deploy the car-part defect classifier to an Azure ML managed online endpoint.

End-to-end SDK v2 flow:
    1. Connect to the workspace
    2. Register the TorchScript model (azureml/model/) as a CUSTOM_MODEL
    3. Register the custom inference environment (conda.yaml)
    4. Create a managed online endpoint (key auth)
    5. Create a "blue" deployment on a CPU SKU and route 100% traffic to it
    6. (Optional) Invoke the endpoint with a local test image

Prerequisites:
    pip install azure-ai-ml azure-identity
    az login

Usage:
    python azureml/deploy_endpoint.py \
        --subscription-id <sub> --resource-group <rg> --workspace <ws>

    # deploy and immediately smoke-test with an image:
    python azureml/deploy_endpoint.py --subscription-id <sub> \
        --resource-group <rg> --workspace <ws> --test-image path/to/car.jpg
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import sys
import tempfile
from pathlib import Path

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

HERE = Path(__file__).resolve().parent


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Deploy defect classifier to an Azure ML online endpoint.")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint-name", default=None, help="Defaults to a unique carparts-<timestamp> name.")
    parser.add_argument("--model-dir", type=Path, default=HERE / "model")
    parser.add_argument("--registered-model", default=None,
                        help="Use an already-registered model by name (e.g. from cloud training) "
                             "instead of uploading --model-dir. Uses the latest version.")
    parser.add_argument("--score-dir", type=Path, default=HERE / "onlinescoring")
    parser.add_argument("--conda-file", type=Path, default=HERE / "conda.yaml")
    parser.add_argument("--instance-type", default="Standard_DS3_v2", help="CPU SKU; sufficient for a small CNN.")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--test-image", type=Path, default=None, help="Optional image to smoke-test after deploy.")
    return parser


def run(args: argparse.Namespace) -> int:
    """Register model + environment and deploy the online endpoint."""
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (
        CodeConfiguration,
        Environment,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
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
            name="pytorch-inf-env",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
            conda_file=str(args.conda_file),
        )
    )
    print(f"   \u2713 {env.name}:{env.version}")

    endpoint_name = args.endpoint_name or "carparts-" + datetime.datetime.now().strftime("%m%d%H%M%f")
    print(f"Creating endpoint '{endpoint_name}'...")
    ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=endpoint_name, description="Car-part defect classifier", auth_mode="key")
    ).result()

    print("Creating 'blue' deployment (this can take several minutes)...")
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=model.id,
        environment=env.id,
        code_configuration=CodeConfiguration(code=str(args.score_dir), scoring_script="score.py"),
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    print("Routing 100% traffic to 'blue'...")
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    print("\n" + "=" * 60)
    print("Deployment complete.")
    print(f"   Endpoint : {endpoint_name}")
    print(f"   Scoring  : {endpoint.scoring_uri}")
    print(f"   Key      : {keys.primary_key}")
    print("=" * 60)

    if args.test_image:
        if not args.test_image.exists():
            print(f"\nWarning: test image not found: {args.test_image}", file=sys.stderr)
        else:
            _smoke_test(ml_client, endpoint_name, args.test_image)

    return EXIT_SUCCESS


def _smoke_test(ml_client, endpoint_name: str, image_path: Path) -> None:
    """Invoke the endpoint with a local image and print the prediction."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"image_base64": b64}, f)
        request_file = f.name
    print(f"\nInvoking endpoint with {image_path.name}...")
    result = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name, deployment_name="blue", request_file=request_file
    )
    print(f"Response: {result}")


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
