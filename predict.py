import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
    return tf.keras.models.load_model(
        "fer_final.keras",
        compile=False
    )

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