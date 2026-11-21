
import os
import json
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATASET_PATH = Path("dataset")

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15

os.makedirs("model", exist_ok=True)

# ==========================================
# 2. FIND DATASET FOLDERS AUTOMATICALLY
# ==========================================

EXPECTED_CLASSES = {
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
}


def find_split_folder(split_name):

    for folder in DATASET_PATH.rglob("*"):

        if not folder.is_dir():
            continue

        parent_names = [
            parent.name.lower()
            for parent in folder.parents
        ]

        if not any(
            name.startswith(split_name)
            for name in parent_names
        ):
            continue

        child_classes = {
            item.name.lower()
            for item in folder.iterdir()
            if item.is_dir()
        }

        if EXPECTED_CLASSES.issubset(child_classes):
            return folder

    raise FileNotFoundError(
        f"Could not find {split_name} dataset folder. "
        "Please check your dataset structure."
    )


TRAIN_PATH = find_split_folder("train")
VALIDATION_PATH = find_split_folder("validation")
TEST_PATH = find_split_folder("test")

print("Dataset folders found successfully!")
print("Training:", TRAIN_PATH)
print("Validation:", VALIDATION_PATH)
print("Testing:", TEST_PATH)

# ==========================================
# 3. LOAD TRAINING DATASET
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

class_names = train_dataset.class_names

print("\nDetected fruit categories:")
print(class_names)

if len(class_names) != 6:
    raise ValueError("Expected six fruit categories.")

# ==========================================
# 4. LOAD VALIDATION DATASET
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

# ==========================================
# 5. LOAD TEST DATASET
# ==========================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

# ==========================================
# 6. OPTIMIZE DATA LOADING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ==========================================
# 7. IMAGE DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1)

])

# ==========================================
# 8. BUILD CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(shape=(160, 160, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(
        32, (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        64, (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        128, (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )

])

# ==========================================
# 9. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nCNN Model Summary:")
model.summary()

# ==========================================
# 10. EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)

# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nStarting CNN Model Training...")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[early_stopping]

)

# ==========================================
# 12. EVALUATE MODEL
# ==========================================

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("\n========== MODEL RESULTS ==========")

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

model.save("model/fruit_freshness.keras")

with open("model/class_names.json", "w") as file:

    json.dump(class_names, file)

print("\nModel saved successfully!")

# ==========================================
# 14. PLOT ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Fruit Freshness Prediction - Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("model/accuracy_graph.png")

plt.show()

print("\nTraining completed successfully!")