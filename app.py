import streamlit as st
from PIL import Image
import pandas as pd

from predict import (
    predict_emotion,
    CLASS_NAMES
)

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Facial Emotion Recognition")

st.markdown(
    """
Upload a face image and predict the emotion.
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    emotion, confidence, probs = predict_emotion(
        image
    )

    st.success(
        f"Predicted Emotion: {emotion}"
    )

    st.info(
        f"Confidence: {confidence:.2%}"
    )

    df = pd.DataFrame({
        "Emotion": CLASS_NAMES,
        "Probability": probs
    })

    st.subheader(
        "Emotion Probabilities"
    )

    st.bar_chart(
        df.set_index("Emotion")
    )