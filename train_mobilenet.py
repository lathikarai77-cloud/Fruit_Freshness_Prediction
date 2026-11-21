
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =====================================
# 1. CONFIGURATION
# =====================================

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs("model", exist_ok=True)

# =====================================
# 2. DATASET PATHS
# =====================================

train_path = (
    "dataset/train-20230708T062807Z-001/train"
)

validation_path = (
    "dataset/validation-20230708T062810Z-001/validation"
)

test_path = (
    "dataset/test-20230708T062801Z-001/test"
)

# =====================================
# 3. LOAD TRAINING DATA
# =====================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

class_names = train_ds.class_names

print("Fruit categories:", class_names)

# =====================================
# 4. LOAD VALIDATION DATA
# =====================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False
)

# =====================================
# 5. LOAD TEST DATA
# =====================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False
)

# =====================================
# 6. PERFORMANCE OPTIMIZATION
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =====================================
# 7. DATA AUGMENTATION
# =====================================

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10)
])

# =====================================
# 8. LOAD MOBILENETV2
# =====================================

print("\nLoading MobileNetV2 pretrained model...")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =====================================
# 9. BUILD CLASSIFICATION MODEL
# =====================================

inputs = layers.Input(shape=(160, 160, 3))

x = augmentation(inputs)

# MobileNetV2 preprocessing: [0,255] to [-1,1]
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# =====================================
# 10. COMPILE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMobileNetV2 Model Summary:")
model.summary()

# =====================================
# 11. CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/best_mobilenet.keras",
    monitor="val_loss",
    save_best_only=True
)

# =====================================
# 12. TRAIN MODEL
# =====================================

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# =====================================
# 13. LOAD BEST MODEL
# =====================================

best_model = tf.keras.models.load_model(
    "model/best_mobilenet.keras"
)

# =====================================
# 14. TEST MODEL
# =====================================

print("\nEvaluating test dataset...")

test_loss, test_accuracy = best_model.evaluate(
    test_ds
)

print("\n========== RESULTS ==========")

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

# =====================================
# 15. SAVE MODEL AND LABELS
# =====================================

best_model.save("model/mobilenet_fruit.keras")

with open("model/mobilenet_class_names.json", "w") as f:
    json.dump(class_names, f)

# =====================================
# 16. ACCURACY GRAPH
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Fruit Freshness Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("model/mobilenet_accuracy.png")

plt.show()

print("\nMobileNetV2 training completed!")