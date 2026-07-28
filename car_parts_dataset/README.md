# Car Parts Quality Dataset

## Overview
This dataset contains 780 synthetic close-up images of individual car parts
photographed on the assembly line, labeled by surface quality. The goal is to
assess whether a set of parts can compose a solid, damage-free car.

## Dataset Structure
```
car_parts_dataset/
├── train/                          # Training images (80%)
│   ├── no_defect/
│   ├── minor_defect/
│   ├── major_defect/
├── val/                            # Validation images (10%)
│   ├── no_defect/
│   ├── minor_defect/
│   ├── major_defect/
├── test/                           # Test images (10%)
│   ├── no_defect/
│   ├── minor_defect/
│   ├── major_defect/
├── annotations/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── metadata.json
└── README.md
```

## Image Properties
- **Resolution**: 256x256 pixels
- **Format**: JPEG (quality: 95)
- **Parts**: 13 (door, hood, bumper, fender, windshield, wheel, side_mirror, trunk, headlight, whole_car, whole_car_no_mechanical, roof_top_inside, roof_top_outside)
- **Quality classes**: 3 (no_defect, minor_defect, major_defect)

## Dataset Statistics
- **Total images**: 780
- **Images per part per class**: 20
- **Training set**: 80%
- **Validation set**: 10%
- **Test set**: 10%

## Quality Class Definitions
- **no_defect**: Pristine, factory-fresh part with a flawless surface, ready to assemble.
- **minor_defect**: Exactly one clearly visible but small, localized defect (a single scratch, dent, paint chip, or scuff) on an otherwise flawless surface.
- **major_defect**: Multiple large, obvious surface defects spread across a wide area (scratches, dents, peeling paint, rust) but still intact and not broken.

## Usage

### Loading with PyTorch
```python
from torchvision import transforms
from torch.utils.data import DataLoader, ImageFolder

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("car_parts_dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Loading with pandas (CSV)
```python
import pandas as pd

train_df = pd.read_csv("car_parts_dataset/annotations/train.csv")
val_df = pd.read_csv("car_parts_dataset/annotations/val.csv")
test_df = pd.read_csv("car_parts_dataset/annotations/test.csv")
```

## Recommended Model Architecture
For image classification with this dataset, consider:
- **ResNet50** - Balanced accuracy and speed
- **EfficientNetB2** - Good for small datasets with transfer learning
- **Vision Transformer (ViT)** - Modern approach, requires more data

## Training Tips
1. Use **transfer learning** (pre-trained on ImageNet) due to small dataset
2. Apply **data augmentation**: rotation, brightness, contrast variations
3. Use **class weights** to handle potential imbalance
4. Monitor validation loss to prevent overfitting
5. Consider **fine-tuning** only the last layers initially

## Metadata CSV Format
Each CSV contains the following columns:
- `image_path`: Relative path to image
- `filename`: Image filename
- `part`: Car part name (door, hood, bumper, ...)
- `quality`: Label (no_defect, minor_defect, major_defect)
- `index`: Sample index within the part/quality group
- `split`: Dataset split (train, val, test)
- `image_size`: Image dimensions

## Generated On
Date: 2026-06-22
