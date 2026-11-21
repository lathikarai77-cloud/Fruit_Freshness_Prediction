
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Dataset folders
base = Path("dataset")

splits = {
    "Training": base / "train-20230708T062807Z-001" / "train",
    "Validation": base / "validation-20230708T062810Z-001" / "validation",
    "Testing": base / "test-20230708T062801Z-001" / "test"
}

categories = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

# Display sample images
fig, axes = plt.subplots(6, 3, figsize=(12, 20))

for row, category in enumerate(categories):

    for col, (split_name, folder) in enumerate(splits.items()):

        category_folder = folder / category

        images = [
            p for p in category_folder.iterdir()
            if p.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".webp"]
        ]

        if not images:
            axes[row, col].set_title("No images found")
            axes[row, col].axis("off")
            continue

        image_path = random.choice(images)

        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(
            f"{split_name}\n{category}"
        )
        axes[row, col].axis("off")

plt.tight_layout()

plt.savefig("model/dataset_samples.png")

plt.show()

print("Dataset sample images displayed successfully!")