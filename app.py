
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request

# =====================================
# 1. FLASK CONFIGURATION
# =====================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "mobilenet_fruit.keras"

CLASS_PATH = BASE_DIR / "model" / "mobilenet_class_names.json"

IMAGE_SIZE = (160, 160)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

print("Loading Fruit Freshness Model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("Model loaded successfully!")
print("Fruit categories:", class_names)

# =====================================
# 3. CHECK IMAGE EXTENSION
# =====================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# =====================================
# 4. PREDICTION FUNCTION
# =====================================

def predict_fruit(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    # Identify fruit type
    if "apple" in predicted_class:
        fruit_name = "Apple"

    elif "banana" in predicted_class:
        fruit_name = "Banana"

    elif "orange" in predicted_class:
        fruit_name = "Orange"

    else:
        fruit_name = "Unknown"

    # Identify freshness
    if predicted_class.startswith("fresh"):
        freshness = "Fresh"

    elif predicted_class.startswith("rotten"):
        freshness = "Rotten"

    else:
        freshness = "Unknown"

    return fruit_name, freshness, confidence

# =====================================
# 5. HOME PAGE
# =====================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    uploaded_image = None

    if request.method == "POST":

        if "fruit_image" not in request.files:

            error = "Please select an image."

        else:

            file = request.files["fruit_image"]

            if file.filename == "":

                error = "Please select an image."

            elif not allowed_file(file.filename):

                error = "Only JPG, JPEG, PNG and WEBP images are allowed."

            else:

                try:

                    image_data = file.read()

                    image = Image.open(
                        BytesIO(image_data)
                    )

                    image.load()

                    uploaded_image = (
                        "data:image/jpeg;base64,"
                        + __import__("base64").b64encode(
                            BytesIO(
                                _image_to_jpeg(image)
                            ).getvalue()
                        ).decode("utf-8")
                    )

                    fruit, freshness, confidence = predict_fruit(
                        image
                    )

                    result = {
                        "fruit": fruit,
                        "freshness": freshness,
                        "confidence": round(confidence, 2)
                    }

                except (UnidentifiedImageError, OSError):

                    error = "Invalid image. Please upload a valid image."

    return render_template(
        "index.html",
        result=result,
        error=error,
        uploaded_image=uploaded_image
    )


def _image_to_jpeg(image):

    output = BytesIO()

    image.convert("RGB").save(
        output,
        format="JPEG"
    )

    return output.getvalue()


# =====================================
# 6. FILE SIZE ERROR
# =====================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        result=None,
        error="Image size must be below 5 MB.",
        uploaded_image=None
    ), 413


# =====================================
# 7. RUN APPLICATION
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )