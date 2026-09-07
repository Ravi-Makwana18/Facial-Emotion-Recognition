import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import json
import tempfile
import zipfile

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]



def _remove_legacy_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for child in value.values():
            _remove_legacy_config(child)
    elif isinstance(value, list):
        for child in value:
            _remove_legacy_config(child)


def _load_model():
    model_path = Path(__file__).with_name("fer_cnn.keras")
    with zipfile.ZipFile(model_path) as source:
        config = json.loads(source.read("config.json"))
        _remove_legacy_config(config)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as compatible_model:
            with zipfile.ZipFile(compatible_model, "w") as target:
                for item in source.infolist():
                    content = json.dumps(config).encode() if item.filename == "config.json" else source.read(item)
                    target.writestr(item, content)
            compatible_path = compatible_model.name

        try:
            return tf.keras.models.load_model(compatible_path, compile=False)
        finally:
            Path(compatible_path).unlink(missing_ok=True)


model = _load_model()


def predict_emotion(image):

    image = image.convert("L")

    image = image.resize((48, 48))

    image = np.array(image)

    image = image.reshape(1, 48, 48, 1)

    image = image.astype("float32")

    preds = model.predict(image, verbose=0)[0]

    predicted_emotion = CLASS_NAMES[np.argmax(preds)]

    confidence = float(np.max(preds))

    return predicted_emotion, confidence, preds