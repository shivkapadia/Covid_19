import streamlit as st
import requests
import json
from io import BytesIO

st.title('COVID Prediction')

image_file = st.file_uploader("Upload Image")

if image_file is not None:

    image = image_file.getvalue()

    response = requests.post(
        "https://shiv8587-covid-19.hf.space/prediction",
        files = {
            "image": BytesIO(image)
        }
    )

    label = json.loads(response._content)
    st.write(f"Patient is {label['class']}")