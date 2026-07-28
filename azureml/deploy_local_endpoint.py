#!/usr/bin/env python3

"""Deploy the car-part defect classifier to a LOCAL Azure ML endpoint (Docker).

A local endpoint runs the same scoring container as a managed online endpoint. 
It gives you a real http://localhost scoring URL for testing.

End-to-end SDK v2 flow (all with local=True):
    1. Connect to the workspace (used only for asset metadata)
    2. Create a local online endpoint
    3. Create a local "blue" deployment from local model + score.py + conda env
       (this builds a Docker image and starts the container)
    4. (Optional) Invoke the local endpoint with a test image

Prerequisites:
    pip install azure-ai-ml azure-identity
    Docker Desktop running
    az login

Usage:
    python azureml/deploy_local_endpoint.py \
        --subscription-id <sub> --resource-group <rg> --workspace <ws> \
        --model-dir _joblogs_v7/artifacts/outputs/model \
        --test-image car_parts_dataset/test/no_defect/<some>.jpg
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

HERE = Path(__file__).resolve().parent


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Deploy defect classifier to a LOCAL Azure ML endpoint.")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint-name", default="carparts-local")
    parser.add_argument("--deployment-name", default="blue")
    parser.add_argument("--model-dir", type=Path, default=HERE / "model",
                        help="Local folder containing model.pt + labels.json.")
    parser.add_argument("--score-dir", type=Path, default=HERE / "onlinescoring")
    parser.add_argument("--conda-file", type=Path, default=HERE / "conda.yaml")
    parser.add_argument("--test-image", type=Path, default=None, help="Optional image to smoke-test after deploy.")
    return parser


def run(args: argparse.Namespace) -> int:
    """Create a local endpoint + deployment and (optionally) smoke-test it."""
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

    model_path = args.model_dir / "model.pt"
    if not model_path.exists():
        print(f"Error: {model_path} not found.", file=sys.stderr)
        return EXIT_ERROR

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    print(f"Creating LOCAL endpoint '{args.endpoint_name}'...")
    ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=args.endpoint_name, description="Car-part defect classifier (local)"),
        local=True,
    )

    print(f"Creating LOCAL deployment '{args.deployment_name}' (builds a Docker image, can take a few minutes)...")
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=Model(path=str(args.model_dir), type=AssetTypes.CUSTOM_MODEL, name="car-parts-pytorch"),
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
            conda_file=str(args.conda_file),
        ),
        code_configuration=CodeConfiguration(code=str(args.score_dir), scoring_script="score.py"),
        instance_type="Standard_DS3_v2",  # ignored for local, but required by the schema
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment, local=True)

    endpoint = ml_client.online_endpoints.get(args.endpoint_name, local=True)
    print("\n" + "=" * 60)
    print("Local endpoint running.")
    print(f"   Endpoint : {args.endpoint_name}")
    print(f"   Scoring  : {endpoint.scoring_uri}")
    print("=" * 60)

    if args.test_image:
        if not args.test_image.exists():
            print(f"\nWarning: test image not found: {args.test_image}", file=sys.stderr)
        else:
            _smoke_test(ml_client, args.endpoint_name, args.deployment_name, args.test_image)

    print("\nTo stop and remove the local endpoint later, run:")
    print(f"   ml_client.online_endpoints.begin_delete(name='{args.endpoint_name}', local=True)")
    return EXIT_SUCCESS


def _smoke_test(ml_client, endpoint_name: str, deployment_name: str, image_path: Path) -> None:
    """Invoke the local endpoint with a local image and print the prediction."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"image_base64": b64}, f)
        request_file = f.name
    print(f"\nInvoking local endpoint with {image_path.name}...")
    result = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name, deployment_name=deployment_name, request_file=request_file, local=True
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
