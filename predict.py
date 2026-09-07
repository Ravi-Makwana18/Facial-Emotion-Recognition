import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

@st.cache_resource
def load_model():
    dense_from_config = tf.keras.layers.Dense.from_config

    def compatible_dense_from_config(config):
        config = dict(config)
        config.pop("quantization_config", None)
        return dense_from_config(config)

    tf.keras.layers.Dense.from_config = compatible_dense_from_config
    try:
        return tf.keras.models.load_model(
            Path(__file__).with_name("fer_cnn.h5"),
            compile=False
        )
    finally:
        tf.keras.layers.Dense.from_config = dense_from_config

model = load_model()


def predict_emotion(image):

    image = image.convert("L")

    image = image.resize((48, 48))

    image = np.array(image)

    image = image.reshape(1, 48, 48, 1)

    image = image.astype("float32")

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_class = CLASS_NAMES[
        np.argmax(predictions)
    ]

    confidence = float(
        np.max(predictions)
    )

    return (
        predicted_class,
        confidence,
        predictions
    )