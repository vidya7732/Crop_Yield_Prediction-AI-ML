import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌾 Smart Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f0f9ff, #cbebff);
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #2c7a7b;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("Crop_Yield_Pickle_File_Saved.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🌾 Smart Crop Yield Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI + Machine Learning based Precision Farming</div>", unsafe_allow_html=True)
st.divider()

# ---------------- FARMER INFO ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("👨‍🌾 Farmer Information")
farmer_name = st.text_input("Farmer Name *")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INPUT SECTIONS ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🌱 Soil & Climate")

    soil_pH = st.number_input("Soil pH", 0.0, 14.0, 6.5, step=0.10)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 2000.0, 800.0, step=0.10)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.10)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, step=0.10)
    soil_health = st.slider("Soil Health Score", 0.0, 10.0, 7.0, step=0.10)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚜 Farming Practices")

    land_area = st.number_input("Land Area (acres)", 0.10, 100.0, 2.0, step=0.10)
    fertilizer_qty = st.number_input("Fertilizer Quantity (kg)", 0.0, 500.0, 120.0, step=0.10)
    pesticide = st.number_input("Pesticide Used (liters)", 0.0, 100.0, 10.0, step=0.10)

    region = st.selectbox("Region", ["East", "North", "South", "West"])
    crop = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean"])
    fert_type = st.selectbox("Fertilizer Type", ["DAP", "NPK", "Urea", "Organic"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

if st.button("🔍 Predict Crop Yield", use_container_width=True):

    if not farmer_name.strip():
        st.warning("⚠️ Farmer Name is required")
        st.stop()

    region_map = {"East":[1,0,0,0],"North":[0,1,0,0],"South":[0,0,1,0],"West":[0,0,0,1]}
    crop_map = {
        "Rice":[1,0,0,0,0,0],"Wheat":[0,1,0,0,0,0],
        "Maize":[0,0,1,0,0,0],"Cotton":[0,0,0,1,0,0],
        "Sugarcane":[0,0,0,0,1,0],"Soybean":[0,0,0,0,0,1]
    }
    fert_map = {"DAP":[1,0,0,0],"NPK":[0,1,0,0],"Urea":[0,0,1,0],"Organic":[0,0,0,1]}

    fertilizer_efficiency = fertilizer_qty / land_area
    temperature_stress = abs(temperature - 25)
    moisture_index = (rainfall * humidity) / 100
    region_weighted_yield = land_area * rainfall

    features = [
        soil_pH, rainfall, fertilizer_qty, pesticide,
        temperature, humidity, land_area,
        *region_map[region], *crop_map[crop], *fert_map[fert_type],
        soil_health, fertilizer_efficiency,
        temperature_stress, moisture_index, region_weighted_yield
    ]

    prediction = model.predict(np.array([features]))[0]

    st.success("✅ Prediction Completed Successfully")
    st.metric("🌾 Predicted Yield (Tonnes)", f"{prediction:.2f}")

    # ---------------- CONFIDENCE ----------------
    stress_score = temperature_stress + (fertilizer_efficiency / 50)
    confidence = max(40, 100 - stress_score * 10)
    st.progress(int(confidence))
    st.metric("Confidence Score (%)", f"{confidence:.1f}%")

    # ---------------- GRAPH ----------------
    st.subheader("📊 Yield Comparison Analysis")

    ideal_yield = land_area * 8
    improvement = ideal_yield - prediction

    yield_df = pd.DataFrame({
        "Yield Type": ["Predicted Yield", "Ideal Yield"],
        "Yield (Tonnes)": [prediction, ideal_yield]
    })

    st.bar_chart(yield_df.set_index("Yield Type"))

    st.markdown(
        f"### 🟢 **+{improvement:.2f} tonnes possible improvement** if inputs are optimized"
    )

    # ---------------- SUGGESTIONS ----------------
    st.subheader("🌱 Smart Improvement Suggestions")

    suggestions = []

    if soil_pH < 6 or soil_pH > 7.5:
        suggestions.append("✔ Maintain soil pH between 6.0 – 7.5 using lime or organic matter")

    if fertilizer_efficiency < 50:
        suggestions.append("✔ Optimize fertilizer usage per acre for better nutrient absorption")

    if temperature_stress > 5:
        suggestions.append("✔ Consider heat-resistant crop varieties or better irrigation")

    if rainfall < 600:
        suggestions.append("✔ Improve irrigation or rainwater harvesting")

    if soil_health < 6:
        suggestions.append("✔ Increase organic manure & crop rotation for soil health")

    if not suggestions:
        suggestions.append("✅ Current inputs are near optimal. Maintain best practices!")

    for s in suggestions:
        st.write(s)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🚀 Deployed using Streamlit | Machine Learning Powered Agriculture</center>",
    unsafe_allow_html=True
)
