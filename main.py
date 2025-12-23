from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import os

app = FastAPI(
    title="Crop Yield Prediction API",
    description="ML model deployed using FastAPI",
    version="1.0"
)

# ===================== MODEL LOAD (FIXED) =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Crop_Yield_Pickle_File_Saved.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===================== INPUT SCHEMA =====================
class CropInput(BaseModel):
    soil_pH: float
    rainfall_mm: float
    fertilizer_quantity_kg: float
    pesticide_used_liters: float
    temperature_c: float
    humidity_percent: float
    land_area_acres: float = Field(gt=0)
    region: str
    crop_type: str
    fertilizer_type: str
    soil_health_score: float

# ===================== ROUTES =====================
@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running"}

@app.post("/predict")
def predict(data: CropInput):

    region_map = {
        "East": [1, 0, 0, 0],
        "North": [0, 1, 0, 0],
        "South": [0, 0, 1, 0],
        "West": [0, 0, 0, 1]
    }

    crop_map = {
        "Rice": [1, 0, 0, 0, 0, 0],
        "Wheat": [0, 1, 0, 0, 0, 0],
        "Maize": [0, 0, 1, 0, 0, 0],
        "Cotton": [0, 0, 0, 1, 0, 0],
        "Sugarcane": [0, 0, 0, 0, 1, 0],
        "Soybean": [0, 0, 0, 0, 0, 1]
    }

    fert_map = {
        "DAP": [1, 0, 0, 0],
        "NPK": [0, 1, 0, 0],
        "Urea": [0, 0, 1, 0],
        "Organic": [0, 0, 0, 1]
    }

    if data.region not in region_map:
        raise HTTPException(400, "Invalid region")

    if data.crop_type not in crop_map:
        raise HTTPException(400, "Invalid crop type")

    if data.fertilizer_type not in fert_map:
        raise HTTPException(400, "Invalid fertilizer type")

    fertilizer_efficiency = data.fertilizer_quantity_kg / data.land_area_acres
    temperature_stress = abs(data.temperature_c - 25)
    moisture_index = (data.rainfall_mm * data.humidity_percent) / 100
    region_weighted_yield = data.land_area_acres * data.rainfall_mm

    features = [
        data.soil_pH,
        data.rainfall_mm,
        data.fertilizer_quantity_kg,
        data.pesticide_used_liters,
        data.temperature_c,
        data.humidity_percent,
        data.land_area_acres,
        *region_map[data.region],
        *crop_map[data.crop_type],
        *fert_map[data.fertilizer_type],
        data.soil_health_score,
        fertilizer_efficiency,
        temperature_stress,
        moisture_index,
        region_weighted_yield
    ]

    prediction = model.predict(np.array([features]))

    return {
        "predicted_yield_tonnes": round(float(prediction[0]), 2)
    }
