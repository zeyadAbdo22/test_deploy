from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
import kagglehub

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224
DR_CLASSES = {0: "DR", 1: "No_DR"}  

DR_model = None

def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def load_and_prepare_image(file: UploadFile):
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return preprocess_image(img)

def make_prediction(model, img_array, class_labels, binary=False):
    prediction = model.predict(img_array)
    if binary:
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])
    else:
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

    label = class_labels[predicted_class]
    return label, confidence, prediction.tolist()

def load_DR_model():
    global DR_model 
    path = kagglehub.model_download("zeyadabdo/diabetic-retinopathy-models/keras/vgg16-v1")
    DR_model_path = os.path.join(path, "Diabetic-Retinopathy-vgg16-model.h5")
    DR_model = load_model(DR_model_path, compile=False)   
    
@app.on_event("startup")
async def load_models():
    try:
        load_DR_model()
        print("DR model loaded.")
    except Exception as e:
        print(f"Failed to load DR model: {e}")

@app.get("/")
def root():
    return {"message": "Multi-Disease Detection API is running!"}

@app.post("/DR")
async def predict_DR(file: UploadFile = File(...)):
    try:
        img_array = load_and_prepare_image(file)
        label, confidence, raw = make_prediction(DR_model, img_array, DR_CLASSES ,binary=True)
        return {
            "success": True,
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw": raw
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
