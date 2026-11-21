
import json
import tensorflow as tf
import numpy as np

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Load saved model
model = tf.keras.models.load_model(
    "model/fruit_freshness.keras"
)

# Load class names
with open("model/class_names.json", "r") as file:
    class_names = json.load(file)

print("Fruit Categories:", class_names)

# Test dataset location
test_path = Path(
    "dataset/test-20230708T062801Z-001/test"
)

# Load test images
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(160, 160),
    batch_size=32,
    class_names=class_names,
    shuffle=False
)

# Actual labels
y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])

# Predicted labels
predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)

# Class distribution
print("\nActual test image count:", len(y_true))

print("\nClass-wise test image count:")

for i, name in enumerate(class_names):
    print(name, ":", np.sum(y_true == i))

# Confusion matrix
print("\nCONFUSION MATRIX")

print(confusion_matrix(y_true, y_pred))

# Classification report
print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))