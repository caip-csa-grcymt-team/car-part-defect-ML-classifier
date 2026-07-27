"""
Car Assembly Dataset Generator
Generates synthetic car assembly images with damage labels and multiple angles.
Organizes them in an ML-ready folder structure.

Requirements:
    - requests (for API calls)
    - Pillow (PIL)
    - opencv-python
    - pandas
    - numpy
    - json (built-in)

Usage:
    1. Set your API key and configuration below
    2. Run: python car_assembly_dataset_generator.py
"""

import os
import sys
import json
import csv
import base64
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import io

# Ensure console output can render Unicode (checkmarks, emoji) on Windows
# terminals that default to a legacy code page such as cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Configuration
CONFIG = {
    "output_dir": "car_parts_dataset",
    "image_size": (256, 256),  # (width, height); model trains at 224, so 256 gives crop headroom
    # Individual car parts photographed on the assembly line.
    "parts": [
        "door",
        "hood",
        "bumper",
        "fender",
        "windshield",
        "wheel",
        "side_mirror",
        "trunk",
        "headlight",
        # Whole-vehicle and roof-focused categories.
        "whole_car",
        "whole_car_no_mechanical",
        "roof_top_inside",
        "roof_top_outside",
    ],
    # Quality classes (defect severity). Goal: decide whether a set of parts
    # can compose a solid, defect-free car. All parts are structurally intact
    # (not broken); only their surface quality differs across the three tiers.
    "quality_classes": ["no_defect", "minor_defect", "major_defect"],
    "images_per_class": 20,  # images per part per quality class
    # Number of images generated concurrently. gpt-image API calls are network
    # (I/O) bound, so a thread pool gives a large speedup. Keep this at or below
    # the deployment's requests-per-minute limit to avoid throttling (429s).
    "max_workers": 2,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "seed": 42,
}

# Image generation backend configuration.
# SECURITY: Prefer setting the API key via the AZURE_OPENAI_API_KEY environment
# variable instead of hardcoding it. The fallback value below should be rotated
# in Azure AI Foundry once you confirm generation works.
API_CONFIG = {
    "provider": "azure_gpt_image",  # Options: "azure_gpt_image", "local_simulation"
    "endpoint": os.environ.get(
        "AZURE_OPENAI_IMAGE_ENDPOINT",
        "https://aif-centre-swe.cognitiveservices.azure.com/openai/deployments/gpt-image-2/images/generations",
    ),
    "api_version": "2024-02-01",
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
    "model": "gpt-image-2",
    # gpt-image models support 1024x1024, 1024x1536, 1536x1024, or "auto".
    "generation_size": "1024x1024",
    "quality": "medium",  # low | medium | high | auto
    # Retry transient network/API failures (DNS blips, dropped connections)
    # before giving up on an image. No local placeholder is ever produced.
    "max_retries": 4,
}


class CarAssemblyDatasetGenerator:
    """Generate synthetic car-part images labeled by damage quality."""

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.image_size = config["image_size"]
        self.parts = config["parts"]
        self.quality_classes = config["quality_classes"]
        
        # Set random seed for reproducibility
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        
        # Create directory structure
        self._create_directory_structure()

    def _create_directory_structure(self):
        """Create the dataset folder structure."""
        # Create main directories: split / quality_class
        for split in ["train", "val", "test"]:
            for quality_class in self.quality_classes:
                split_dir = self.output_dir / split / quality_class
                split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create annotations directory
        annotations_dir = self.output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Directory structure created: {self.output_dir}")

    def generate_synthetic_image_azure(
        self,
        part: str,
        quality: str,
        idx: int
    ) -> Image.Image:
        """
        Generate a realistic car-part image using the Azure OpenAI gpt-image-2
        deployment. Retries on transient network/API failures (e.g. DNS or
        connection blips) with exponential backoff, then raises if all attempts
        fail - there is no local fallback, so only real generated photos enter
        the dataset.
        Note: Requires a valid API key/endpoint in API_CONFIG.
        """
        import time
        import requests

        prompt = self._create_prompt(part, quality, idx)

        headers = {
            "Content-Type": "application/json",
            "api-key": API_CONFIG["api_key"],
        }

        payload = {
            "model": API_CONFIG["model"],
            "prompt": prompt,
            "n": 1,
            "size": API_CONFIG["generation_size"],
            "quality": API_CONFIG["quality"],
        }

        max_retries = API_CONFIG.get("max_retries", 4)
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    API_CONFIG["endpoint"],
                    headers=headers,
                    params={"api-version": API_CONFIG["api_version"]},
                    json=payload,
                    timeout=300,
                )

                # Rate limited: honor the server's Retry-After cooldown (in
                # seconds) instead of the short exponential backoff, since the
                # deployment may ask for a much longer wait (e.g. 39s).
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait = int(float(retry_after)) if retry_after else 2 ** attempt
                    except (TypeError, ValueError):
                        wait = 2 ** attempt
                    wait = max(wait, 1) + 1  # small buffer
                    last_error = RuntimeError(
                        f"API error 429: {response.text[:200]}"
                    )
                    if attempt < max_retries:
                        print(
                            f"      rate limited (429); waiting {wait}s "
                            f"(attempt {attempt}/{max_retries - 1})..."
                        )
                        time.sleep(wait)
                        continue
                    raise last_error

                if response.status_code != 200:
                    raise RuntimeError(
                        f"API error {response.status_code}: {response.text[:300]}"
                    )

                data = response.json()["data"][0]
                # gpt-image models return base64-encoded image content.
                if data.get("b64_json"):
                    img_bytes = base64.b64decode(data["b64_json"])
                    img = Image.open(io.BytesIO(img_bytes))
                else:
                    img_response = requests.get(data["url"], timeout=60)
                    img = Image.open(io.BytesIO(img_response.content))

                img = img.convert("RGB")
                # Resize to the configured dataset resolution.
                if img.size != tuple(self.image_size):
                    img = img.resize(self.image_size, Image.LANCZOS)
                return img

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s, 8s, ...
                    print(
                        f"      retry {attempt}/{max_retries - 1} after error "
                        f"({type(e).__name__}); waiting {wait}s..."
                    )
                    time.sleep(wait)

        # All attempts exhausted - signal failure to the caller (no fallback).
        raise RuntimeError(last_error)

    def _create_prompt(self, part: str, quality: str, idx: int) -> str:
        """Create a detailed, photorealistic prompt for a single car part."""
        # Human-friendly descriptions and key inspection features per part.
        part_desc = {
            "door": "a single car door panel",
            "hood": "a single car hood (bonnet) panel",
            "bumper": "a single car bumper",
            "fender": "a single car fender (wing) panel",
            "windshield": "a single car windshield glass panel",
            "wheel": "a single car wheel with tire and alloy rim",
            "side_mirror": "a single car side mirror assembly",
            "trunk": "a single car trunk lid (tailgate) panel",
            "headlight": "a single car headlight assembly",
            "whole_car": "a complete, fully assembled car shown in full, the entire vehicle visible",
            "whole_car_no_mechanical": "a complete car body shell (exterior bodywork and panels) without any mechanical components such as the engine, drivetrain or wheels installed",
            "roof_top_inside": "the roof of a car viewed from inside the cabin, showing the interior headliner ceiling",
            "roof_top_outside": "the roof of a car viewed from outside, showing the exterior top roof panel of the vehicle",
        }

        quality_desc = {
            "no_defect": (
                "in PRISTINE, factory-fresh, completely defect-free condition: a flawless, "
                "perfectly even surface with immaculate paint or finish, uniform gloss, "
                "absolutely no dents, no scratches, no scuffs, no discoloration, and clean "
                "crisp edges, ready to be assembled onto a car"
            ),
            "minor_defect": (
                "with EXACTLY ONE single, clearly visible but small and localized cosmetic "
                "defect confined to one spot on the surface, while the rest of the part stays "
                "clean and glossy: for example one distinct dark scratch line, one obvious "
                "dent with a visible shadow, one chipped-paint spot showing bare material, or "
                "one scuff mark. The single flaw is plainly noticeable at a glance and stands "
                "out clearly against the surrounding flawless surface, but it stays small and "
                "isolated to a single area, NOT spread across the part. The part keeps its "
                "perfect overall shape"
            ),
            "major_defect": (
                "with MULTIPLE LARGE, obvious surface defects that are immediately visible "
                "from a distance, but the part is still intact and not broken: several deep "
                "and long scratches, prominent dents, large peeling or chipped paint patches, "
                "heavy scuffing, rust spots and clear discoloration spread across a wide area "
                "of the surface. The damage is severe and unmistakable. The part keeps its "
                "original overall shape, but its surface is heavily and obviously degraded"
            ),
        }

        # Per-image variation so the ~20 images per class are diverse rather than
        # near-duplicates. idx deterministically selects a viewpoint, lighting and
        # background combination, which spreads the class across realistic conditions.
        viewpoints = [
            "front-facing straight-on view", "three-quarter angle view",
            "slightly elevated top-down angle", "low side-angle view",
            "head-on centered view", "gentle diagonal perspective",
        ]
        lightings = [
            "bright even factory inspection lighting", "cool white LED workshop lighting",
            "soft diffused overhead lighting", "neutral daylight-balanced studio lighting",
            "crisp directional inspection lamp lighting",
        ]
        backgrounds = [
            "a clean industrial workbench", "a metal parts rack",
            "a neutral grey inspection table", "a matte factory-floor surface",
            "a stainless-steel quality-control station",
        ]
        viewpoint = viewpoints[idx % len(viewpoints)]
        lighting = lightings[idx % len(lightings)]
        background = backgrounds[idx % len(backgrounds)]

        part_text = part_desc.get(part, f"a single car {part.replace('_', ' ')}")

        prompt = (
            f"A photorealistic, high-resolution close-up product photograph of {part_text}, "
            f"{quality_desc[quality]}. "
            f"Shown from a {viewpoint}, resting on {background} under {lighting}, "
            f"at a car manufacturing assembly line or quality-inspection station. "
            f"Single isolated component centered in frame, sharp focus, accurate automotive "
            f"materials and reflections, neutral industrial background, "
            f"shot like a real DSLR quality-control inspection photo, "
            f"no text, no watermark, no illustration, only one part visible."
        )

        return prompt

    def generate_dataset(self):
        """Generate the complete dataset with concurrent API calls."""
        total = len(self.parts) * len(self.quality_classes) * self.config["images_per_class"]
        max_workers = self.config.get("max_workers", 6)
        print(f"\n🚀 Generating {total} car-part images...")
        print(f"   Parts: {self.parts}")
        print(f"   Quality classes: {self.quality_classes}")
        print(f"   Resolution: {self.image_size[0]}x{self.image_size[1]}")
        print(f"   Parallel workers: {max_workers}\n")

        # Build the full task list up front. Split assignment and the image
        # counter are decided here in deterministic (seeded) order so results
        # stay reproducible regardless of the order tasks finish in the pool.
        tasks = []
        image_counter = 0
        for part in self.parts:
            for quality in self.quality_classes:
                for idx in range(self.config["images_per_class"]):
                    rand = random.random()
                    if rand < self.config["train_split"]:
                        split = "train"
                    elif rand < self.config["train_split"] + self.config["val_split"]:
                        split = "val"
                    else:
                        split = "test"
                    filename = f"{part}_{quality}_{idx:02d}_{image_counter:04d}.jpg"
                    filepath = self.output_dir / split / quality / filename
                    tasks.append({
                        "part": part,
                        "quality": quality,
                        "idx": idx,
                        "split": split,
                        "filename": filename,
                        "filepath": filepath,
                    })
                    image_counter += 1

        metadata = []
        skipped = 0
        completed = 0
        resumed = 0

        # Resume-aware: any image already on disk (from a prior interrupted run)
        # is kept as-is and only recorded in metadata, so re-running fills only
        # the gaps left by transient outages instead of regenerating everything.
        pending_tasks = []
        for task in tasks:
            if task["filepath"].exists():
                resumed += 1
                metadata.append({
                    "image_path": str(task["filepath"].relative_to(self.output_dir)),
                    "filename": task["filename"],
                    "part": task["part"],
                    "quality": task["quality"],
                    "index": task["idx"],
                    "split": task["split"],
                    "image_size": self.image_size,
                })
            else:
                pending_tasks.append(task)

        if resumed:
            print(f"↩️  Resuming: {resumed} existing image(s) kept, "
                  f"{len(pending_tasks)} still to generate.\n")

        # Generate images concurrently. The API call is network-bound, so a
        # thread pool overlaps the waiting time across many requests. Each PIL
        # image is saved in this main thread as its future resolves, which keeps
        # file writes and the metadata list free of race conditions.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.generate_synthetic_image_azure,
                    task["part"], task["quality"], task["idx"],
                ): task
                for task in pending_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                try:
                    img = future.result()
                except Exception as e:
                    skipped += 1
                    print(f"   ✗ [{completed}/{len(pending_tasks)}] Skipped "
                          f"{task['part']}/{task['quality']} #{task['idx']}: {e}")
                    continue

                filepath = task["filepath"]
                img.save(str(filepath), quality=95)

                metadata.append({
                    "image_path": str(filepath.relative_to(self.output_dir)),
                    "filename": task["filename"],
                    "part": task["part"],
                    "quality": task["quality"],
                    "index": task["idx"],
                    "split": task["split"],
                    "image_size": self.image_size,
                })
                print(f"   ✓ [{completed}/{len(pending_tasks)}] {task['filename']}")

        if skipped:
            print(f"\n⚠️  Skipped {skipped} image(s): gpt-image API was unavailable (no placeholder saved).")

        return metadata

    def create_csv_splits(self, metadata: List[Dict]):
        """Create train/val/test CSV files."""
        train_data = [m for m in metadata if m["split"] == "train"]
        val_data = [m for m in metadata if m["split"] == "val"]
        test_data = [m for m in metadata if m["split"] == "test"]
        
        # Save train split
        self._save_csv(train_data, "train.csv")
        # Save validation split
        self._save_csv(val_data, "val.csv")
        # Save test split
        self._save_csv(test_data, "test.csv")
        
        print(f"\n📊 Dataset Split Summary:")
        print(f"   Training images:   {len(train_data)}")
        print(f"   Validation images: {len(val_data)}")
        print(f"   Test images:       {len(test_data)}")
        print(f"   Total images:      {len(metadata)}")

    def _save_csv(self, data: List[Dict], filename: str):
        """Save metadata to CSV file."""
        if not data:
            return
        
        filepath = self.output_dir / "annotations" / filename
        
        with open(filepath, "w", newline="") as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(data)
        
        print(f"   ✓ {filename}")

    def save_metadata_json(self, metadata: List[Dict]):
        """Save complete metadata to JSON."""
        metadata_path = self.output_dir / "metadata.json"
        
        with open(metadata_path, "w") as f:
            json.dump({
                "config": self.config,
                "total_images": len(metadata),
                "images": metadata
            }, f, indent=2)
        
        print(f"   ✓ metadata.json")

    def create_readme(self):
        """Create README documentation."""
        readme_path = self.output_dir / "README.md"

        total = len(self.parts) * len(self.quality_classes) * self.config["images_per_class"]
        parts_tree = "\n".join(
            f"│   ├── {q}/" for q in self.quality_classes
        )

        readme_content = f"""# Car Parts Quality Dataset

## Overview
This dataset contains {total} synthetic close-up images of individual car parts
photographed on the assembly line, labeled by surface quality. The goal is to
assess whether a set of parts can compose a solid, damage-free car.

## Dataset Structure
```
{self.output_dir.name}/
├── train/                          # Training images (80%)
{parts_tree}
├── val/                            # Validation images (10%)
{parts_tree}
├── test/                           # Test images (10%)
{parts_tree}
├── annotations/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── metadata.json
└── README.md
```

## Image Properties
- **Resolution**: {self.image_size[0]}x{self.image_size[1]} pixels
- **Format**: JPEG (quality: 95)
- **Parts**: {len(self.parts)} ({", ".join(self.parts)})
- **Quality classes**: {len(self.quality_classes)} ({", ".join(self.quality_classes)})

## Dataset Statistics
- **Total images**: {total}
- **Images per part per class**: {self.config['images_per_class']}
- **Training set**: {self.config['train_split']*100:.0f}%
- **Validation set**: {self.config['val_split']*100:.0f}%
- **Test set**: {self.config['test_split']*100:.0f}%

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
"""
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print(f"   ✓ README.md")

    def run(self):
        """Run the complete dataset generation pipeline."""
        print("="*60)
        print("🚗 CAR ASSEMBLY DATASET GENERATOR")
        print("="*60)
        
        # Generate all images
        metadata = self.generate_dataset()
        
        print(f"\n📁 Organizing dataset...")
        # Create CSV splits
        self.create_csv_splits(metadata)
        
        # Save metadata
        print(f"\n💾 Saving metadata...")
        self.save_metadata_json(metadata)
        
        # Create documentation
        print(f"\n📖 Creating documentation...")
        self.create_readme()
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset generation complete!")
        print(f"📂 Location: {self.output_dir.absolute()}")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    generator = CarAssemblyDatasetGenerator(CONFIG)
    generator.run()


if __name__ == "__main__":
    main()
