from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load model files from your local disk
MODEL_DIR = os.path.expanduser("~/Documents/fastapi_app")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model_tfidf.pkl")
MLB_PATH = os.path.join(MODEL_DIR, "mlb.pkl")

pipeline = None
mlb = None

@app.on_event("startup")
def load_models():
    global pipeline, mlb
    try:
        pipeline = joblib.load(MODEL_PATH)
        mlb = joblib.load(MLB_PATH)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")

# Test route
@app.get("/")
def home():
    return {"message": "API is running locally!"}

# Request body structure
class InputData(BaseModel):
    text: str

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    if not pipeline or not mlb:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        input_text = data.text
        prediction = pipeline.predict([input_text])
        tags = mlb.inverse_transform(prediction)
        return {"tags": tags[0] if tags else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
