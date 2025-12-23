import streamlit as st
import requests

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🌾 Crop Yield Prediction System",
    page_icon="🌾",
    layout="centered"
)

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center;'>🌾 Crop Yield Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-powered yield estimation for smart farming</p>",
    unsafe_allow_html=True
)

st.divider()

# ================= FARMER DETAILS =================
st.subheader("👨‍🌾 Farmer Information")

farmer_name = st.text_input(
    "Farmer Name *",
    placeholder="Enter farmer full name"
)

# ================= SOIL & WEATHER =================
st.subheader("🌱 Soil & Weather Parameters")

col1, col2 = st.columns(2)

with col1:
    soil_pH = st.number_input(
        "Soil pH (Ideal: 6.5 – 7.5)",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=150.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=30.0
    )

    land_area = st.number_input(
        "Land Area (acres)",
        min_value=0.1,
        value=3.0
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    fertilizer_qty = st.number_input(
        "Fertilizer Quantity (kg)",
        min_value=0.0,
        value=100.0
    )

    pesticide = st.number_input(
        "Pesticide Used (liters)",
        min_value=0.0,
        value=1.5
    )

    soil_health = st.number_input(
        "Soil Health Score (1–10)",
        min_value=0.0,
        max_value=10.0,
        value=8.5
    )

# ================= FARM PRACTICES =================
st.subheader("🚜 Farming Practices")

region = st.selectbox(
    "Region",
    ["East", "North", "South", "West"],
    index=0
)

crop = st.selectbox(
    "Crop Type",
    ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean"],
    index=5
)

fert_type = st.selectbox(
    "Fertilizer Type",
    ["DAP", "NPK", "Urea", "Organic"],
    index=2
)

st.divider()

# ================= PREDICTION =================
predict_btn = st.button("🔍 Predict Crop Yield", use_container_width=True)

if predict_btn:

    if farmer_name.strip() == "":
        st.warning("⚠️ Farmer Name is required.")
        st.stop()

    payload = {
        "farmer_name": farmer_name,
        "soil_pH": soil_pH,
        "rainfall_mm": rainfall,
        "fertilizer_quantity_kg": fertilizer_qty,
        "pesticide_used_liters": pesticide,
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "land_area_acres": land_area,
        "soil_health_score": soil_health,
        "region": region,
        "crop_type": crop,
        "fertilizer_type": fert_type
    }

    try:
        with st.spinner("🔄 Predicting crop yield using ML model..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=10
            )

        if response.status_code == 200:
            result = response.json()
            predicted_yield = result["predicted_yield_tonnes"]

            st.success("✅ Prediction Successful")

            st.metric(
                "🌾 Predicted Yield (Tonnes)",
                f"{predicted_yield:.2f}"
            )

            st.caption(f"👨‍🌾 Farmer: **{farmer_name}**")

            # ================= EXPLANATION =================
            st.markdown("### 🧠 How this yield was predicted?")
            st.info(
                "The machine learning model analyzed soil quality, weather conditions, "
                "land area, fertilizer usage, and crop type. "
                "Higher soil health, adequate rainfall, optimal fertilizer quantity, "
                "and suitable crop selection contributed to increased predicted yield."
            )

        else:
            st.error(f"❌ Backend Error | Status Code: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Backend server not running. Please start FastAPI first.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:13px;'>"
    "Powered by Machine Learning | Crop Yield Prediction System"
    "</p>",
    unsafe_allow_html=True
)
