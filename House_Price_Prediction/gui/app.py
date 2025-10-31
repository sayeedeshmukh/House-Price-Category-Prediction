import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

st.set_page_config(page_title="House Price Category Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Category Predictor")
st.caption("California Housing Dataset — Predict Low / Medium / High")

best_model_path = MODELS_DIR / "best_model.pkl"
label_enc_path = MODELS_DIR / "label_encoder.pkl"

if not best_model_path.exists() or not label_enc_path.exists():
    st.error("Models not found. Please run training first.")
    st.stop()

model = joblib.load(best_model_path)
label_encoder = joblib.load(label_enc_path)

# Sensible default ranges derived from typical data ranges
st.sidebar.header("Input Features")
MedInc = st.sidebar.slider("Median Income (10k USD)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
HouseAge = st.sidebar.slider("House Age (years)", min_value=1.0, max_value=52.0, value=20.0, step=1.0)
AveRooms = st.sidebar.slider("Average Rooms", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
AveBedrms = st.sidebar.slider("Average Bedrooms", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
Population = st.sidebar.slider("Population", min_value=1.0, max_value=5000.0, value=800.0, step=10.0)
AveOccup = st.sidebar.slider("Average Occupancy", min_value=1.0, max_value=7.0, value=3.0, step=0.1)
Latitude = st.sidebar.slider("Latitude", min_value=32.0, max_value=42.0, value=35.0, step=0.1)
Longitude = st.sidebar.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)

X_df = pd.DataFrame(
    [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
    columns=FEATURE_COLUMNS,
)

st.subheader("Your Inputs")
st.dataframe(X_df, use_container_width=True)

if st.button("Predict Price Category", type="primary"):
    pred = model.predict(X_df)[0]
    label = label_encoder.inverse_transform([pred])[0]

    color = {
        "Low": "#e74c3c",
        "Medium": "#f1c40f",
        "High": "#2ecc71",
    }.get(label, "#3498db")

    st.markdown(
        f"<div style='padding:16px;border-radius:8px;background:{color};color:white;text-align:center;font-size:24px;'>Prediction: <b>{label}</b></div>",
        unsafe_allow_html=True,
    )

st.info("Tip: Adjust the sliders on the left and click Predict.")




