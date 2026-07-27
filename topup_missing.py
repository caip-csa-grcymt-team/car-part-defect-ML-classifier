"""
Top-Up Missing Car-Part Images

Scans the existing car_parts_dataset for how many images exist per
(part, quality) pair, then generates ONLY the missing slots needed to reach
the configured images_per_class target. Images skipped by earlier runs
(e.g. due to intermittent DNS/network failures) are filled in here without
regenerating what already succeeded.

After generation, the train/val/test CSVs and metadata.json are rebuilt from
ALL images currently on disk so the annotations stay consistent.

Usage:
    python topup_missing.py
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

# Reuse the generator's configuration and image-generation logic so prompts,
# API handling, retries, and resizing stay identical to the original run.
from car_assembly_dataset_generator import (
    CONFIG,
    CarAssemblyDatasetGenerator,
)


def count_existing_per_class(output_dir: Path, parts: List[str], qualities: List[str]) -> Dict[str, int]:
    """Count existing .jpg files per '{part}|{quality}' key across all splits."""
    counts: Dict[str, int] = {f"{p}|{q}": 0 for p in parts for q in qualities}
    for jpg in output_dir.rglob("*.jpg"):
        name = jpg.name
        for part in parts:
            for quality in qualities:
                if name.startswith(f"{part}_{quality}_"):
                    counts[f"{part}|{quality}"] += 1
                    break
    return counts


def max_existing_counter(output_dir: Path) -> int:
    """Return the highest 4-digit counter used in existing filenames, or -1."""
    highest = -1
    for jpg in output_dir.rglob("*.jpg"):
        stem = jpg.stem  # e.g. door_no_defect_02_0086
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            highest = max(highest, int(parts[1]))
    return highest


def rebuild_annotations(generator: CarAssemblyDatasetGenerator) -> List[Dict]:
    """Rescan every image on disk and rebuild CSV splits + metadata.json."""
    output_dir = generator.output_dir
    parts = generator.parts
    qualities = generator.quality_classes

    metadata: List[Dict] = []
    for split in ["train", "val", "test"]:
        for jpg in sorted((output_dir / split).rglob("*.jpg")):
            name = jpg.name
            matched_part = None
            matched_quality = None
            for part in parts:
                for quality in qualities:
                    if name.startswith(f"{part}_{quality}_"):
                        matched_part = part
                        matched_quality = quality
                        break
                if matched_part:
                    break
            if not matched_part:
                continue

            # index is the 2-digit field right after the quality label.
            prefix = f"{matched_part}_{matched_quality}_"
            remainder = jpg.stem[len(prefix):]  # e.g. 02_0086
            idx_str = remainder.split("_", 1)[0]
            index = int(idx_str) if idx_str.isdigit() else 0

            metadata.append({
                "image_path": str(jpg.relative_to(output_dir)),
                "filename": name,
                "part": matched_part,
                "quality": matched_quality,
                "index": index,
                "split": split,
                "image_size": list(generator.image_size),
            })

    generator.create_csv_splits(metadata)
    generator.save_metadata_json(metadata)
    return metadata


def main() -> None:
    print("=" * 60)
    print("🩹 CAR-PART DATASET TOP-UP (fill missing slots only)")
    print("=" * 60)

    generator = CarAssemblyDatasetGenerator(CONFIG)
    output_dir = generator.output_dir
    parts = generator.parts
    qualities = generator.quality_classes
    target = CONFIG["images_per_class"]

    counts = count_existing_per_class(output_dir, parts, qualities)
    counter = max_existing_counter(output_dir) + 1

    # Report and compute the work list.
    todo: List[tuple] = []  # (part, quality, how_many_missing)
    print("\nCurrent coverage (existing / target):")
    for part in parts:
        line = f"  {part:12s}"
        for quality in qualities:
            have = counts[f"{part}|{quality}"]
            missing = max(0, target - have)
            if missing:
                todo.append((part, quality, missing))
            line += f"  {quality}={have}/{target}"
        print(line)

    total_missing = sum(m for _, _, m in todo)
    if total_missing == 0:
        print("\n✅ Nothing missing — dataset already complete. Rebuilding annotations only.")
        rebuild_annotations(generator)
        print("\n✅ Done.")
        return

    print(f"\n🚀 Generating {total_missing} missing image(s)...\n")

    generated = 0
    skipped = 0
    for part, quality, missing in todo:
        print(f"📦 {part} / {quality}: need {missing} more")
        # Continue idx numbering after the existing count for this pair.
        start_idx = counts[f"{part}|{quality}"]
        for offset in range(missing):
            idx = start_idx + offset
            try:
                img = generator.generate_synthetic_image_azure(part, quality, idx)
            except Exception as e:
                skipped += 1
                print(f"   ✗ Skipped {part}/{quality} #{idx}: {e}")
                continue

            # Assign split with the same probabilities as the main run.
            rand = random.random()
            if rand < CONFIG["train_split"]:
                split = "train"
            elif rand < CONFIG["train_split"] + CONFIG["val_split"]:
                split = "val"
            else:
                split = "test"

            filename = f"{part}_{quality}_{idx:02d}_{counter:04d}.jpg"
            filepath = output_dir / split / quality / filename
            img.save(str(filepath), quality=95)
            counter += 1
            generated += 1
            print(f"   ✓ {filename}")

    print(f"\n🖼️  Generated {generated} new image(s); {skipped} still failed.")

    print("\n📁 Rebuilding annotations from all images on disk...")
    metadata = rebuild_annotations(generator)

    print(f"\n{'=' * 60}")
    print(f"✅ Top-up complete! Total images now: {len(metadata)}")
    print(f"📂 Location: {output_dir.absolute()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
