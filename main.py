from fastapi import FastAPI, UploadFile, File
import json
from PIL import Image
from io import BytesIO
import numpy as np
from model import build_model

app = FastAPI()

# Load model
image_shape = (256, 256, 1)
num_classes = 3
model = build_model(image_shape, num_classes)
model.load_weights('./model_with_weights.h5')
classes = {
    0: 'COVID-19',
    1: 'Non-COVID',
    2: 'Normal'
}

@app.get("/")
def first_api():
    return {
        "response": "COVID Prediction"
    }

@app.post("/prediction")
async def prediction(image: UploadFile = File(...)):
    image = await image.read()

    # process image
    image = Image.open(BytesIO(image))
    image = image.resize((image_shape[0], image_shape[1]))
    image = image.convert('L')
    image = np.expand_dims(image, axis=2)

    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    label = np.argmax(prediction, axis=-1).tolist()

    return {
        "label": label,
        "class": classes[label]
    }