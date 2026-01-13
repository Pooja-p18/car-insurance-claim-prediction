"""
Car Insurance Claim Prediction - Streamlit App
==============================================
Clean UI + SAFE preprocessing using trained pipeline
"""

import streamlit as st
import pandas as pd
import pickle
from src.preprocessing import InsurancePreprocessor

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Insurance Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 🔶 CUSTOM CSS (ORANGE PREDICT BUTTON - FIXED)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Normal buttons */
    div.stButton > button {
        background-color: #ff8c00 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        height: 3em !important;
        border: none !important;
    }

    /* Form submit button (THIS WAS MISSING) */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #ff8c00 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        height: 3em !important;
        border: none !important;
        width: 100% !important;
    }

    /* Hover effect */
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #e67600 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# LOAD MODEL & PREPROCESSOR
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = InsurancePreprocessor.load("models/preprocessor.pkl")
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return preprocessor, model


preprocessor, model = load_artifacts()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Single Prediction", "📊 Batch Prediction", "ℹ️ About"]
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🚗 Car Insurance Claim Prediction System")
st.markdown("Predict claim probability using machine learning")
st.markdown("---")

# =============================================================================
# SINGLE PREDICTION PAGE
# =============================================================================
if page == "🔮 Single Prediction":

    st.header("🔮 Single Customer Prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Policy")
            policy_tenure = st.number_input("Policy Tenure (months)", 1, 120, 12)
            age_of_policyholder = st.slider("Policyholder Age", 18, 80, 35)

            st.subheader("📍 Location")
            area_cluster = st.selectbox("Area Cluster", ["C1", "C2", "C3", "C4"])
            population_density = st.number_input("Population Density", 0, 10000, 500)

        with col2:
            st.subheader("🚙 Vehicle")
            age_of_car = st.slider("Car Age (years)", 0, 20, 3)
            segment = st.selectbox("Segment", ["A", "B1", "B2", "C1", "C2"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

            st.subheader("⚙️ Engine")
            displacement = st.number_input("Displacement (cc)", 500, 5000, 1500)
            max_power = st.number_input("Max Power", 30, 500, 100)
            max_torque = st.number_input("Max Torque", 50, 800, 200)

        with col3:
            st.subheader("🛡️ Safety")
            airbags = st.selectbox("Airbags", [0, 1, 2, 4, 6, 8])
            ncap_rating = st.slider("NCAP Rating", 0, 5, 3)

            is_esc = st.checkbox("ESC")
            is_brake_assist = st.checkbox("Brake Assist")
            is_parking_sensors = st.checkbox("Parking Sensors")
            is_parking_camera = st.checkbox("Parking Camera")

        # 🔶 ORANGE BUTTON
        submit = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submit:
        try:
            input_data = {
                "policy_tenure": policy_tenure,
                "age_of_car": age_of_car,
                "age_of_policyholder": age_of_policyholder,
                "area_cluster": area_cluster,
                "population_density": population_density,
                "segment": segment,
                "fuel_type": fuel_type,
                "max_torque": max_torque,
                "max_power": max_power,
                "displacement": displacement,
                "airbags": airbags,
                "ncap_rating": ncap_rating,
                "is_esc": int(is_esc),
                "is_brake_assist": int(is_brake_assist),
                "is_parking_sensors": int(is_parking_sensors),
                "is_parking_camera": int(is_parking_camera),
            }

            input_df = pd.DataFrame([input_data])

            # SAFE TRANSFORMATION
            processed = preprocessor.transform(input_df)

            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0]

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            colA, colB, colC = st.columns(3)

            with colA:
                st.success("LOW RISK" if pred == 0 else "HIGH RISK")

            with colB:
                st.metric("Claim Probability", f"{prob[1]*100:.2f}%")

            with colC:
                st.metric("No-Claim Probability", f"{prob[0]*100:.2f}%")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)

# =============================================================================
# BATCH PREDICTION PAGE
# =============================================================================
elif page == "📊 Batch Prediction":

    st.header("📊 Batch Prediction")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} records")
        st.dataframe(df.head())

        if st.button("🔮 Predict"):
            preds, probs = [], []

            for _, row in df.iterrows():
                row_df = pd.DataFrame([row.to_dict()])
                processed = preprocessor.transform(row_df)
                preds.append(model.predict(processed)[0])
                probs.append(model.predict_proba(processed)[0][1])

            df["prediction"] = preds
            df["claim_probability"] = probs

            st.success("Prediction complete")
            st.dataframe(df)

            st.download_button(
                "📥 Download Results",
                df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

# =============================================================================
# ABOUT PAGE
# =============================================================================
else:
    st.header("ℹ️ About")
    st.markdown("""
    ### 🚗 Car Insurance Claim Prediction
    - Trained on 58k+ records
    - Robust preprocessing pipeline
    - Handles missing & unseen features safely
    - Production-ready ML deployment

    **Tech Stack**
    - Python, Pandas, Scikit-learn
    - LightGBM / XGBoost
    - Streamlit
    """)
