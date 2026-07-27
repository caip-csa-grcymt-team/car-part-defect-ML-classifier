

# Car Assembly Dataset - Quick Start and Usage Guide


This project trains a three-class image classifier that inspects car parts and labels each image as one of:

- `no_defect` — the part looks clean.
- `minor_defect` — one small, localized flaw (scratch, chip, small dent).
- `major_defect` — widespread or severe damage.

The current best model reaches about 84% accuracy on the held-out test set. It is essentially perfect on `major_defect` and mainly errs on the cautious side by flagging some clean parts as `minor_defect`.

## Repository layout

| Path | Purpose |
|------|---------|
| `car_assembly_dataset_generator.py` | Generates the synthetic dataset using Azure OpenAI gpt-image. |
| `train_classifier.py` | Trains and evaluates the classifier. |
| `model/model.pt` + `model/labels.json` | The trained model and its class labels. |
| `azureml/submit_training_job.py` | Submits a training job to Azure Machine Learning. |
| `azureml/deploy_batch_endpoint.py` | Deploys a batch scoring endpoint (recommended for teams). |
| `azureml/deploy_local_endpoint.py` | Runs a local Docker endpoint for offline single-image scoring. |
| `requirements.txt` | Python dependencies. |

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

### 3. Inspect the dataset

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

## Dataset summary

| Property | Value |
|----------|-------|
| Classes | `no_defect`, `minor_defect`, `major_defect` |
| Image size | 256 x 256 pixels |
| Approximate total images | ~714 |
| Split | ~586 train / ~58 val / ~70 test |
| Format | JPEG |

Class counts are roughly balanced. Exact numbers live in `car_parts_dataset/metadata.json`.

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

### Train locally

```powershell
python train_classifier.py --backbone efficientnet_b0 --epochs 50 --img-size 256
```

### Train on Azure Machine Learning

```powershell
python azureml/submit_training_job.py `
  --subscription-id "<sub-id>" `
  --resource-group "<resource-group>" `
  --workspace "<workspace>" `
  --compute-size "Standard_D4ds_v5" `
  --data-asset "car-parts-images" `
  --backbone efficientnet_b0 `
  --epochs 50
```

## Expected performance

| Class | Behavior |
|-------|----------|
| `major_defect` | Near-perfect; rarely confused with other classes. |
| `minor_defect` | Good; occasionally read as `no_defect`. |
| `no_defect` | Weakest; sometimes flagged as `minor_defect`. |

Overall test accuracy is about 84%. The remaining errors sit almost entirely on the boundary between `minor_defect` and `no_defect`, which is expected because subtle defects are visually close to clean parts.

## Deployment

Two quota-free options are available.

| Option | When to use |
|--------|-------------|
| Batch endpoint (`azureml/deploy_batch_endpoint.py`) | Recommended for teams. Scores a whole folder of images in the shared Azure workspace with no local setup. |
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

## Common pitfalls

- Do not commit API keys. Provide them through environment variables only.
- Use pretrained backbones; training from scratch on this dataset size will not converge well.
- Keep preprocessing identical across train, validation, and test.
- Watch validation loss and keep the best checkpoint rather than the final epoch.

## Next steps

1. Regenerate or extend the dataset if you need more coverage.
2. Retrain and compare against the current ~84% baseline.
3. Deploy the batch endpoint and share it with your team.
4. Score new images and review the per-class results.
