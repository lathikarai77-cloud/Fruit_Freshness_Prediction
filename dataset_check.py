
from pathlib import Path

dataset_path = Path("dataset")

folders = {
    "Training": dataset_path / "train-20230708T062807Z-001" / "train",
    "Validation": dataset_path / "validation-20230708T062810Z-001" / "validation",
    "Testing": dataset_path / "test-20230708T062801Z-001" / "test"
}

extensions = {".jpg", ".jpeg", ".png", ".webp"}

for split_name, folder in folders.items():

    print("\n==============================")
    print(split_name.upper())
    print("==============================")

    total = 0

    for category in sorted(folder.iterdir()):

        if category.is_dir():

            count = sum(
                1 for file in category.iterdir()
                if file.suffix.lower() in extensions
            )

            total += count

            print(f"{category.name}: {count} images")

    print("Total images:", total)