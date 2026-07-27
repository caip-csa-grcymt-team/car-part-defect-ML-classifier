"""
Resilient annotation rebuild.

Rescans every image on disk (including the new whole-car / roof subjects) and
rebuilds the train/val/test CSVs and metadata.json. Unlike the generator's
built-in rebuild, this version:

  * includes the extended subject list so the new images are not dropped, and
  * tolerates a locked annotations file (e.g. train.csv still open in Excel):
    the other files are written immediately, the locked file is retried, and if
    it is still locked a side copy (<name>.new.csv) is written instead so no
    work is lost.

Usage:
    python rebuild_annotations_now.py
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from car_assembly_dataset_generator import CONFIG, CarAssemblyDatasetGenerator
from generate_car_views import NEW_SUBJECTS


def scan_metadata(output_dir: Path, parts: List[str], qualities: List[str]) -> List[Dict]:
    """Rescan every image on disk and build metadata rows."""
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
                "image_size": list(CONFIG["image_size"]),
            })
    return metadata


def write_csv(rows: List[Dict], filepath: Path, retries: int = 3) -> Path:
    """Write rows to filepath. On PermissionError, retry, then fall back to a
    <name>.new.csv side file so no data is lost. Returns the path written."""
    if not rows:
        return filepath
    fieldnames = list(rows[0].keys())
    for attempt in range(1, retries + 1):
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return filepath
        except PermissionError:
            if attempt < retries:
                time.sleep(1.5)
                continue
            fallback = filepath.with_suffix(".new.csv")
            with open(fallback, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return fallback


def main() -> None:
    print("=" * 60)
    print("🧾 RESILIENT ANNOTATION REBUILD")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    parts = list(CONFIG["parts"]) + NEW_SUBJECTS
    qualities = CONFIG["quality_classes"]

    metadata = scan_metadata(output_dir, parts, qualities)

    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for row in metadata:
        splits[row["split"]].append(row)

    print("\n📁 Writing annotation files...")
    locked: List[str] = []
    for split, rows in splits.items():
        target = ann_dir / f"{split}.csv"
        written = write_csv(rows, target)
        if written != target:
            locked.append(f"{split}.csv -> {written.name}")
            print(f"   ⚠️  {target.name} locked; wrote {written.name} instead ({len(rows)} rows)")
        else:
            print(f"   ✓ {target.name} ({len(rows)} rows)")

    # metadata.json (rarely locked)
    meta_path = output_dir / "metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"config": CONFIG, "total_images": len(metadata), "images": metadata}, f, indent=2)
        print(f"   ✓ metadata.json ({len(metadata)} images)")
    except PermissionError:
        alt = output_dir / "metadata.new.json"
        with open(alt, "w", encoding="utf-8") as f:
            json.dump({"config": CONFIG, "total_images": len(metadata), "images": metadata}, f, indent=2)
        locked.append(f"metadata.json -> {alt.name}")
        print(f"   ⚠️  metadata.json locked; wrote {alt.name} instead")

    # Per-subject / per-class summary
    print("\n📊 Dataset Split Summary:")
    print(f"   Training images:   {len(splits['train'])}")
    print(f"   Validation images: {len(splits['val'])}")
    print(f"   Test images:       {len(splits['test'])}")
    print(f"   Total images:      {len(metadata)}")

    by_class: Dict[str, int] = {}
    for row in metadata:
        by_class[row["quality"]] = by_class.get(row["quality"], 0) + 1
    print("\n   Per quality class:")
    for q in qualities:
        print(f"     {q:14s} {by_class.get(q, 0)}")

    new_count = sum(1 for r in metadata if r["part"] in NEW_SUBJECTS)
    print(f"\n   New-subject images included: {new_count}")

    if locked:
        print("\n⚠️  Some files were locked and written as side copies:")
        for item in locked:
            print(f"     {item}")
        print("   Close the file in Excel, then rename the .new copy over the original.")
    else:
        print("\n✅ Annotations rebuilt cleanly.")


if __name__ == "__main__":
    main()
