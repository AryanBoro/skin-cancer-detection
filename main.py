import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = FastAPI(title="Skin Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = {
    0: "Actinic Keratosis (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic Nevi (nv)",
    6: "Vascular Lesion (vasc)",
}

CLASS_INFO = {
    0: {"risk": "Pre-cancerous", "color": "orange"},
    1: {"risk": "Malignant", "color": "red"},
    2: {"risk": "Benign", "color": "green"},
    3: {"risk": "Benign", "color": "green"},
    4: {"risk": "Malignant", "color": "red"},
    5: {"risk": "Benign", "color": "green"},
    6: {"risk": "Benign", "color": "green"},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "efficientnetv2s.h5")

model = None
grad_model = None

def load_model():
    global model, grad_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        print("Warning: No Conv2D layer found, Grad-CAM unavailable.")
    else:
        print(f"Using layer '{last_conv_layer}' for Grad-CAM")
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )

    print("Model loaded successfully.")

def preprocess_image(image_bytes: bytes) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = image.resize((224, 224))
    img_array = np.array(original, dtype=np.float32)
    # The model was trained with custom [-1, 1] scaling
    img_array = (img_array / 255.0 - 0.5) * 2.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(original)

def generate_gradcam(img_array: np.ndarray, original_img: np.ndarray, class_idx: int) -> str:
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (heatmap_colored * 0.4 + original_img * 0.6).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {"message": "Skin Cancer Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        img_array, original_img = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")

    predictions = model.predict(img_array)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx]) * 100

    all_classes = [
        {
            "class": CLASS_NAMES[i],
            "confidence": round(float(predictions[i]) * 100, 2),
            "risk": CLASS_INFO[i]["risk"],
        }
        for i in range(len(CLASS_NAMES))
    ]
    all_classes.sort(key=lambda x: x["confidence"], reverse=True)

    gradcam_image = None
    if grad_model is not None:
        try:
            gradcam_image = generate_gradcam(img_array, original_img, predicted_idx)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    return {
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 2),
        "risk_level": CLASS_INFO[predicted_idx]["risk"],
        "all_classes": all_classes,
        "gradcam": gradcam_image,
    }