import json
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import onnxruntime as ort

# Model already exists inside the base image
MODEL_PATH = "hair_classifier_empty.onnx"

TARGET_SIZE = (200, 200)

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load ONNX model once (Lambda-style)
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        data = resp.read()
    return Image.open(BytesIO(data))


def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.NEAREST)

    x = np.array(img).astype("float32") / 255.0
    x = (x - IMG_MEAN[None, None, :]) / IMG_STD[None, None, :]
    x = np.transpose(x, (2, 0, 1))       # CHW
    x = np.expand_dims(x, axis=0)        # NCHW
    return x


def predict(url):
    img = download_image(url)
    x = preprocess(img)
    out = session.run([output_name], {input_name: x})[0]
    return float(out.squeeze())


def lambda_handler(event, context):
    url = event.get("url")
    if not url:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing url"})}

    score = predict(url)
    return {
        "statusCode": 200,
        "body": json.dumps({"score": score})
    }
