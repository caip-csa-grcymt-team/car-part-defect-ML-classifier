"""
Generate Additional Car-View Images (roof views + whole-car subjects)

Extends the existing car_parts_dataset with three NEW subjects, following the
exact same structure and methodology as the original generator and the
top-up script:

    * roof_top             - the car roof, both exterior and interior views
    * car_body_no_wheels   - the whole car WITHOUT wheels or mechanical parts
    * full_car             - the complete whole car

Each subject is generated across all three quality classes
(no_defect, minor_defect, major_defect) with 4 different camera angles per
class (images_per_class), using the same Azure gpt-image-2 backend, retry
logic, 512x512 resizing, filename scheme, and split probabilities.

Nothing already on disk is regenerated. The script is idempotent: it only
creates the slots that are still missing (so it can be re-run to recover from
transient DNS/network failures). Afterwards the train/val/test CSVs and
metadata.json are rebuilt from ALL images on disk (existing parts + new
subjects) so the annotations stay consistent.

Usage:
    python generate_car_views.py
"""

import copy
import random
from typing import Dict, List

from car_assembly_dataset_generator import (
    CONFIG,
    CarAssemblyDatasetGenerator,
)
from topup_missing import (
    count_existing_per_class,
    max_existing_counter,
    rebuild_annotations,
)

# New subject labels (used in "{label}_{quality}_{idx:02d}_{counter:04d}.jpg"
# and the "part" annotation column). Distinct from the isolated "roof_panel".
NEW_SUBJECTS: List[str] = ["roof_top", "car_body_no_wheels", "full_car"]

# Camera angles per subject index (0..3). For roof_top, indexes 0-1 are
# exterior roof views and 2-3 are interior headliner views (the "both" option).
ROOF_VIEWS = {
    0: ("exterior", "from a high three-quarter angle above the front, looking down onto the roof"),
    1: ("exterior", "from a high angle above and behind, showing the full roof panel from the rear"),
    2: ("interior", "from the driver's seat looking upward at the headliner and roof lining"),
    3: ("interior", "from the rear seats looking up and forward at the interior roof lining"),
}

CAR_ANGLES = {
    0: "in a front three-quarter view with the front and one side clearly visible",
    1: "in a rear three-quarter view with the rear and one side clearly visible",
    2: "in a direct side profile view",
    3: "from a high front angle looking slightly down onto the vehicle",
}


class CarViewsGenerator(CarAssemblyDatasetGenerator):
    """Adds photorealistic prompts for roof views and whole-car subjects."""

    def _quality_text_body(self, quality: str) -> str:
        """Defect description for painted exterior bodywork."""
        return {
            "no_defect": (
                "in pristine, factory-fresh showroom condition with flawless glossy "
                "paint, perfectly even body panels, clean panel gaps, and no dents, "
                "scratches or blemishes anywhere"
            ),
            "minor_defect": (
                "in mostly good condition but with a few subtle, barely noticeable "
                "surface imperfections such as faint hairline scratches, a light scuff "
                "or slight paint unevenness on the bodywork. The vehicle is otherwise "
                "clean and fully intact, its overall shape and panels perfect"
            ),
            "major_defect": (
                "with clearly visible surface defects but still complete and not "
                "wrecked: noticeable scratches, shallow dents, obvious paint blemishes, "
                "uneven finish, scuffs or discoloration across the bodywork. The vehicle "
                "keeps its full original shape and structure, only its surface quality "
                "is clearly imperfect"
            ),
        }[quality]

    def _quality_text_headliner(self, quality: str) -> str:
        """Defect description for the interior roof lining / headliner."""
        return {
            "no_defect": (
                "in pristine, factory-fresh condition with a clean, taut, evenly "
                "colored headliner fabric, no stains, sagging, scuffs or marks"
            ),
            "minor_defect": (
                "in mostly good condition with a few subtle, barely noticeable "
                "imperfections such as a faint mark, a slight scuff or very light "
                "unevenness in the headliner fabric, otherwise clean and intact"
            ),
            "major_defect": (
                "with clearly visible imperfections but still intact: noticeable "
                "stains, scuffs, discoloration or slight sagging of the headliner "
                "fabric, while the roof lining remains complete and in place"
            ),
        }[quality]

    def _create_prompt(self, part: str, quality: str, idx: int) -> str:
        # Fall back to the original per-part prompts for the existing 10 parts.
        if part not in NEW_SUBJECTS:
            return super()._create_prompt(part, quality, idx)

        common_tail = (
            "Photorealistic, high-resolution, sharp focus, accurate automotive "
            "materials and reflections, neutral industrial background under bright "
            "factory inspection lighting, shot like a real DSLR quality-control "
            "inspection photo, no text, no watermark, no illustration."
        )

        if part == "roof_top":
            view_type, angle = ROOF_VIEWS[idx % 4]
            if view_type == "exterior":
                subject = (
                    f"A close-up photograph of the exterior roof panel of a complete "
                    f"modern car, viewed {angle}, {self._quality_text_body(quality)}."
                )
            else:
                subject = (
                    f"A photograph of the interior roof headliner (roof lining) inside "
                    f"a modern car cabin, viewed {angle}, "
                    f"{self._quality_text_headliner(quality)}."
                )
            return f"{subject} {common_tail}"

        if part == "car_body_no_wheels":
            angle = CAR_ANGLES[idx % 4]
            subject = (
                f"A photograph of a complete car body shell {angle}, "
                f"WITHOUT any wheels or tires mounted (empty wheel arches) and with no "
                f"visible engine, suspension, drivetrain or other mechanical parts - "
                f"only the painted exterior body panels, doors, hood, roof and glass. "
                f"The bodywork is {self._quality_text_body(quality)}."
            )
            return f"{subject} {common_tail}"

        # full_car
        angle = CAR_ANGLES[idx % 4]
        subject = (
            f"A photograph of a complete whole car with all body panels, glass and "
            f"wheels mounted, {angle}, {self._quality_text_body(quality)}."
        )
        return f"{subject} {common_tail}"


def main() -> None:
    print("=" * 60)
    print("🚗 CAR-VIEW EXPANSION (roof views + whole-car subjects)")
    print("=" * 60)

    # Build an extended config so the generator (and the annotation rebuild)
    # recognizes the new subjects alongside the original 10 parts.
    extended_config = copy.deepcopy(CONFIG)
    extended_config["parts"] = list(CONFIG["parts"]) + NEW_SUBJECTS

    generator = CarViewsGenerator(extended_config)
    output_dir = generator.output_dir
    qualities = generator.quality_classes
    target = CONFIG["images_per_class"]

    # Count only the NEW subjects; existing parts are left untouched.
    counts = count_existing_per_class(output_dir, NEW_SUBJECTS, qualities)
    counter = max_existing_counter(output_dir) + 1

    todo: List[tuple] = []  # (subject, quality, missing)
    print("\nNew-subject coverage (existing / target):")
    for subject in NEW_SUBJECTS:
        line = f"  {subject:20s}"
        for quality in qualities:
            have = counts[f"{subject}|{quality}"]
            missing = max(0, target - have)
            if missing:
                todo.append((subject, quality, missing))
            line += f"  {quality}={have}/{target}"
        print(line)

    total_missing = sum(m for _, _, m in todo)
    if total_missing == 0:
        print("\n✅ Nothing missing — all new subjects complete. Rebuilding annotations only.")
        metadata = rebuild_annotations(generator)
        print(f"\n✅ Done. Total images now: {len(metadata)}")
        return

    print(f"\n🚀 Generating {total_missing} new image(s)...\n")

    generated = 0
    skipped = 0
    for subject, quality, missing in todo:
        print(f"📦 {subject} / {quality}: need {missing} more")
        start_idx = counts[f"{subject}|{quality}"]
        for offset in range(missing):
            idx = start_idx + offset
            try:
                img = generator.generate_synthetic_image_azure(subject, quality, idx)
            except Exception as e:
                skipped += 1
                print(f"   ✗ Skipped {subject}/{quality} #{idx}: {e}")
                continue

            # Same split probabilities as the original run.
            rand = random.random()
            if rand < CONFIG["train_split"]:
                split = "train"
            elif rand < CONFIG["train_split"] + CONFIG["val_split"]:
                split = "val"
            else:
                split = "test"

            filename = f"{subject}_{quality}_{idx:02d}_{counter:04d}.jpg"
            filepath = output_dir / split / quality / filename
            img.save(str(filepath), quality=95)
            counter += 1
            generated += 1
            print(f"   ✓ {filename}")

    print(f"\n🖼️  Generated {generated} new image(s); {skipped} still failed.")

    print("\n📁 Rebuilding annotations from all images on disk...")
    metadata = rebuild_annotations(generator)

    print(f"\n{'=' * 60}")
    print(f"✅ Expansion complete! Total images now: {len(metadata)}")
    print(f"📂 Location: {output_dir.absolute()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
