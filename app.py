import streamlit as st
from PIL import Image
import pandas as pd

from predict import predict_emotion, CLASS_NAMES

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Facial Emotion Recognition")

st.markdown(
    """
Upload a facial image and predict the emotion.

Supported emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
"""
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    emotion, confidence, probabilities = predict_emotion(image)

    st.success(
        f"Predicted Emotion: {emotion}"
    )

    st.info(
        f"Confidence: {confidence:.2%}"
    )

    df = pd.DataFrame({
        "Emotion": CLASS_NAMES,
        "Probability": probabilities
    })

    st.subheader("Prediction Probabilities")

    st.bar_chart(
        df.set_index("Emotion")
    )