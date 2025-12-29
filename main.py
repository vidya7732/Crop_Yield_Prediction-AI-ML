from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="🌾 Smart Crop Yield Prediction API")

# ---------------- LOAD MODEL ----------------
with open("Crop_Yield_Pickle_File_Saved.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- INPUT SCHEMA ----------------
class CropInput(BaseModel):
    soil_pH: float
    rainfall: float
    fertilizer_qty: float
    pesticide: float
    temperature: float
    humidity: float
    soil_health: float
    land_area: float
    region: str
    crop: str
    fert_type: str

# ---------------- ENCODING MAPS ----------------
region_map = {
    "East":[1,0,0,0], "North":[0,1,0,0],
    "South":[0,0,1,0], "West":[0,0,0,1]
}

crop_map = {
    "Rice":[1,0,0,0,0,0], "Wheat":[0,1,0,0,0,0],
    "Maize":[0,0,1,0,0,0], "Cotton":[0,0,0,1,0,0],
    "Sugarcane":[0,0,0,0,1,0], "Soybean":[0,0,0,0,0,1]
}

fert_map = {
    "DAP":[1,0,0,0], "NPK":[0,1,0,0],
    "Urea":[0,0,1,0], "Organic":[0,0,0,1]
}

# ---------------- API ENDPOINT ----------------
@app.post("/predict")
def predict(data: CropInput):

    fertilizer_efficiency = data.fertilizer_qty / data.land_area
    temperature_stress = abs(data.temperature - 25)
    moisture_index = (data.rainfall * data.humidity) / 100
    region_weighted_yield = data.land_area * data.rainfall

    features = [
        data.soil_pH, data.rainfall, data.fertilizer_qty, data.pesticide,
        data.temperature, data.humidity, data.land_area,
        *region_map[data.region],
        *crop_map[data.crop],
        *fert_map[data.fert_type],
        data.soil_health,
        fertilizer_efficiency,
        temperature_stress,
        moisture_index,
        region_weighted_yield
    ]

    prediction = model.predict(np.array([features]))[0]

    stress_score = temperature_stress + (fertilizer_efficiency / 50)
    confidence = max(40, 100 - stress_score * 10)

    ideal_yield = data.land_area * 8
    improvement = ideal_yield - prediction

    return {
        "prediction": round(float(prediction), 2),
        "confidence": round(confidence, 1),
        "ideal_yield": round(ideal_yield, 2),
        "improvement": round(improvement, 2)
    }
