"""
       python house_price_model.py
       streamlit run gui_app.py
"""

import os
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifact(path: str) -> Dict[str, Any]:
    return joblib.load(path)


def main() -> None:
    st.set_page_config(page_title="California Housing Price Category Predictor", page_icon="🏠", layout="centered")
    st.title("🏠 California Housing Price Category Predictor")
    st.write("Predicts price category (Low / Medium / High) using the best trained model.")

    model_path = os.path.join(os.getcwd(), "best_model.pkl")
    if not os.path.exists(model_path):
        st.error("best_model.pkl not found. Please run 'python house_price_model.py' first to train and save the model.")
        return

    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    feature_names: List[str] = artifact["feature_names"]
    class_labels: List[str] = artifact.get("class_labels", ["Low", "Medium", "High"]) 
    best_model_name: str = artifact.get("best_model_name", "Best Model")

    st.sidebar.header("Model Info")
    st.sidebar.write(f"Selected Model: **{best_model_name}**")
    if "results" in artifact:
        st.sidebar.write("Evaluation snapshot (Accuracy):")
        try:
            res_df = pd.DataFrame(artifact["results"]) [["Model", "Accuracy"]]
            st.sidebar.dataframe(res_df, use_container_width=True, height=240)
        except Exception:
            pass

    st.subheader("Enter Feature Values")
    st.caption("These are the original California Housing features. Adjust as needed and click Predict.")

    # Reasonable UI ranges/defaults for the dataset
    defaults = {
        "MedInc": 5.0,        # median income in tens of thousands
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 35.0,
        "Longitude": -119.0,
    }

    # Ensure features are presented in the saved order
    col1, col2 = st.columns(2)
    inputs: Dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        container = col1 if i % 2 == 0 else col2
        with container:
            if fname == "MedInc":
                val = st.number_input(
                    "MedInc (median income in $10k)",
                    min_value=0.0, max_value=20.0, value=float(defaults.get(fname, 5.0)), step=0.1,
                    help="Median income of block group (in tens of thousands of dollars)."
                )
            elif fname == "HouseAge":
                val = st.number_input(
                    "HouseAge (years)",
                    min_value=0.0, max_value=60.0, value=float(defaults.get(fname, 20.0)), step=1.0,
                    help="Median house age of block group."
                )
            elif fname == "AveRooms":
                val = st.number_input(
                    "AveRooms (avg rooms per household)",
                    min_value=0.5, max_value=20.0, value=float(defaults.get(fname, 5.0)), step=0.1,
                    help="Average number of rooms per household."
                )
            elif fname == "AveBedrms":
                val = st.number_input(
                    "AveBedrms (avg bedrooms per household)",
                    min_value=0.2, max_value=5.0, value=float(defaults.get(fname, 1.0)), step=0.1,
                    help="Average number of bedrooms per household."
                )
            elif fname == "Population":
                val = st.number_input(
                    "Population",
                    min_value=1.0, max_value=10000.0, value=float(defaults.get(fname, 1000.0)), step=10.0,
                    help="Block group population."
                )
            elif fname == "AveOccup":
                val = st.number_input(
                    "AveOccup (avg occupants per household)",
                    min_value=0.5, max_value=10.0, value=float(defaults.get(fname, 3.0)), step=0.1,
                    help="Average number of household members."
                )
            elif fname == "Latitude":
                val = st.number_input(
                    "Latitude",
                    min_value=32.0, max_value=43.0, value=float(defaults.get(fname, 35.0)), step=0.1,
                    help="Geographic latitude of the block group."
                )
            elif fname == "Longitude":
                val = st.number_input(
                    "Longitude",
                    min_value=-125.0, max_value=-113.0, value=float(defaults.get(fname, -119.0)), step=0.1,
                    help="Geographic longitude of the block group."
                )
            else:
                val = st.number_input(fname, value=0.0)
        inputs[fname] = float(val)

    if st.button("Predict", type="primary"):
        try:
            input_df = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)
            pred = pipeline.predict(input_df)[0]
            st.success(f"Predicted Category: {pred}")

            # Show probabilities if available
            try:
                proba = pipeline.predict_proba(input_df)[0]
                proba_df = pd.DataFrame({"Class": class_labels, "Probability": proba}).sort_values("Class")
                st.subheader("Class Probabilities")
                st.bar_chart(proba_df.set_index("Class"))
            except Exception:
                st.caption("This model does not expose prediction probabilities.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with st.expander("About the features"):
        st.markdown(
            "- **MedInc**: Median income in block group (in $10k)\n"
            "- **HouseAge**: Median house age\n"
            "- **AveRooms**: Average rooms per household\n"
            "- **AveBedrms**: Average bedrooms per household\n"
            "- **Population**: Block group population\n"
            "- **AveOccup**: Average occupants per household\n"
            "- **Latitude**: Latitude\n"
            "- **Longitude**: Longitude"
        )


if __name__ == "__main__":
    main()





