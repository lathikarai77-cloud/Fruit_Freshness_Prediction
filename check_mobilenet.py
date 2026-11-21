
import json
import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# Load trained MobileNetV2 model
model = tf.keras.models.load_model(
    "model/mobilenet_fruit.keras"
)

# Load class names
with open(
    "model/mobilenet_class_names.json", "r"
) as file:
    class_names = json.load(file)

print("Fruit Categories:", class_names)

# Test dataset path
test_path = (
    "dataset/test-20230708T062801Z-001/test"
)

# Load test images
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(160, 160),
    batch_size=16,
    class_names=class_names,
    shuffle=False
)

# Get actual labels
y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])

# Predict
predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)

# Display confusion matrix
print("\nCONFUSION MATRIX")

print(confusion_matrix(y_true, y_pred))

# Display classification report
print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("\nMobileNetV2 evaluation completed!")