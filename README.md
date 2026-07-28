# Car Assembly Defect Classifier 

## Generate a synthetic car-part image dataset with Azure OpenAI and train a three-class defect classifier.

## Overview

This project:

1. **Generates** a synthetic dataset of car-part photos using the Azure OpenAI gpt-image API.
2. **Trains** an image classifier that labels each part as `no_defect`, `minor_defect`, or `major_defect`.

The current best model is an EfficientNet-B0 that reaches about 84% accuracy on the held-out test set. 

## Repository layout

| Path | Purpose |
|------|---------|
| `car_assembly_dataset_generator.py` | Generates the synthetic dataset using Azure OpenAI gpt-image. |
| `train_classifier.py` | Trains and evaluates the classifier. |
| `model/model.pt` + `model/labels.json` | The trained model and its class labels. |
| `SETUP_AND_TRAINING_GUIDE.md` | Detailed setup, training, and deployment guide. |
| `azureml/submit_training_job.py` | Submits a training job to Azure Machine Learning. |
| `azureml/deploy_batch_endpoint.py` | Deploys a batch scoring endpoint |
| `azureml/deploy_local_endpoint.py` | Runs a local Docker endpoint for offline single-image scoring. |
| `azureml/score_test_folder.py` | Scores a folder of images against the local endpoint and reports predicted vs. actual. |
| `requirements.txt` | Python dependencies. |
| `car_parts_dataset/` | The generated dataset (images, splits, annotations, metadata). |

## Quick start

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate the dataset (optional)

The repository already ships a trained model, so this step is only needed to regenerate or extend the dataset.

Set your Azure OpenAI credentials as environment variables, then run the generator:

```powershell
$env:AZURE_OPENAI_API_KEY = "<your-key>"
$env:AZURE_OPENAI_IMAGE_ENDPOINT = "<your-endpoint>"
python car_assembly_dataset_generator.py
```

The generator is resume-aware: re-running it fills only missing images and rebuilds the annotation CSV files.

### 3. Train the classifier

```powershell
python train_classifier.py --backbone efficientnet_b0 --epochs 50 --img-size 256
```

See [SETUP_AND_TRAINING_GUIDE.md](SETUP_AND_TRAINING_GUIDE.md) for full training and deployment details.

## Dataset

The generator photographs individual car parts on an assembly line across three quality tiers. Every part is structurally intact; only the surface quality differs.

| Property | Value |
|----------|-------|
| Quality classes | `no_defect`, `minor_defect`, `major_defect` |
| Part categories | door, hood, bumper, fender, windshield, wheel, side_mirror, trunk, headlight, whole_car, whole_car_no_mechanical, roof_top_inside, roof_top_outside |
| Image size | 256 x 256 pixels |
| Target per part per class | 20 images |
| Total images | ~714 |
| Split | roughly 80% train / 10% val / 10% test |
| Format | JPEG |

Exact counts live in `car_parts_dataset/metadata.json`.

### Dataset structure

```text
car_parts_dataset/
├── train/
│   ├── no_defect/
│   ├── minor_defect/
│   └── major_defect/
├── val/
├── test/
├── annotations/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── metadata.json
```

Each annotation CSV has the columns: `image_path`, `filename`, `part`, `quality`, `index`, `split`, `image_size`.

## Model and training

The classifier uses transfer learning on an ImageNet-pretrained backbone.

| Setting | Value |
|---------|-------|
| Backbone | EfficientNet-B0 (pretrained) |
| Input resolution | 256 x 256 |
| Strategy | Freeze backbone, fine-tune the last 2 feature blocks |
| Head learning rate | 1e-3 |
| Backbone learning rate | 1e-4 |
| Loss | Cross-entropy with class weights and label smoothing (0.1) |
| Schedule | Cosine with linear warmup |
| Test-time augmentation | Horizontal-flip averaging |

### Expected performance

| Class | Behavior |
|-------|----------|
| `major_defect` | Near-perfect; rarely confused with other classes. |
| `minor_defect` | Good; occasionally read as `no_defect`. |
| `no_defect` | Weakest; sometimes flagged as `minor_defect`. |

Overall test accuracy is about 84%. The remaining errors sit almost entirely on the boundary between `minor_defect` and `no_defect`, which is expected because subtle defects are visually close to clean parts.

## Deployment

Two options are available.

| Option | When to use |
|--------|-------------|
| Batch endpoint (`azureml/deploy_batch_endpoint.py`) | Scores a whole folder of images in the shared Azure workspace with no local setup. |
| Local Docker endpoint (`azureml/deploy_local_endpoint.py`) | Offline, single-image predictions on your own machine. Requires Docker Desktop. |

### Score a folder with the batch endpoint

```python
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

ml = MLClient(DefaultAzureCredential(), "<sub-id>", "<resource-group>", "<workspace>")
job = ml.batch_endpoints.invoke(
    endpoint_name="carparts-batch",
    input=Input(type=AssetTypes.URI_FOLDER, path="folder_of_images"),
)
print(job.name)
```

The job produces a `predictions.csv` with the predicted class and per-class probabilities for every image.

### Get predictions from the local endpoint

After deploying the local endpoint with `azureml/deploy_local_endpoint.py`, score a whole folder (for example the test split) and compare predictions against the known labels:

```powershell
python azureml/score_test_folder.py `
    --subscription-id <sub-id> --resource-group <resource-group> --workspace <workspace> `
    --images-dir car_parts_dataset/test
```

The subfolder name (`major_defect`, `minor_defect`, `no_defect`) is used as the ground-truth label, so the script prints predicted vs. actual per image, writes `test_predictions.csv`, and reports overall accuracy. Point `--images-dir` at any folder of new, unlabelled images to get defect-type predictions for them instead.

## API configuration

Image generation uses the Azure OpenAI gpt-image API. Configure it through environment variables rather than hardcoding secrets:

```python
API_CONFIG = {
    "provider": "azure_gpt_image",  # "azure_gpt_image" or "local_simulation"
    "endpoint": os.environ.get("AZURE_OPENAI_IMAGE_ENDPOINT", ""),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
    "model": "gpt-image-2",         # Azure OpenAI deployment name
}
```

## System requirements

| Tier | Requirements |
|------|--------------|
| Minimum | Python 3.8+, 4 GB RAM, ~1 GB disk. |
| Recommended for training | Python 3.10+, 8+ GB RAM, an NVIDIA/CUDA GPU, ~2 GB disk. |

## Common pitfalls

- Use pretrained backbones; training from scratch on this dataset size will not converge well.
- Keep preprocessing identical across train, validation, and test.
- Watch validation loss and keep the best checkpoint rather than the final epoch.

## Resources

- [PyTorch](https://pytorch.org/docs/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Transfer learning](https://cs231n.github.io/transfer-learning/)
- [Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/)
