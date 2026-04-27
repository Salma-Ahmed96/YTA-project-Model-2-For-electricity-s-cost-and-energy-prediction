import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Electricity Predictor", layout="centered")

# This ensures the app finds files relative to this script's location on the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multi_output_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.joblib")

# --- 2. CORE FUNCTIONS ---

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """Loads the Keras model and the StandardScaler. Uses cache to prevent reloading on every click."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at: {scaler_path}")
        st.stop()
    
    model = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_input(new_data_df, scaler):
    """Preprocesses user input to match the training data format."""
    processed_df = new_data_df.copy()

    # One-hot encode: Season
    if 'season' in processed_df.columns:
        processed_df['season_winter'] = (processed_df['season'].str.lower() == 'winter').astype(int)
        processed_df = processed_df.drop(columns=['season'])
    else:
        processed_df['season_winter'] = 0

    # One-hot encode: Insulation Quality
    if 'insulation_quality' in processed_df.columns:
        processed_df['insulation_quality_low'] = (processed_df['insulation_quality'].str.lower() == 'low').astype(int)
        processed_df['insulation_quality_medium'] = (processed_df['insulation_quality'].str.lower() == 'medium').astype(int)
        processed_df = processed_df.drop(columns=['insulation_quality'])
    else:
        processed_df['insulation_quality_low'] = 0
        processed_df['insulation_quality_medium'] = 0

    # Ensure all training columns are present in the correct order
    expected_feature_columns = [
        'number_of_air_conditioners', 'ac_power_hp', 'number_of_refrigerators',
        'number_of_televisions', 'number_of_fans', 'number_of_computers',
        'average_daily_usage_hours', 'house_size_m2', 'has_water_heater',
        'washing_machine_usage_per_week', 'season_winter', 
        'insulation_quality_low', 'insulation_quality_medium'
    ]

    for col in expected_feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    processed_df = processed_df[expected_feature_columns]
    
    # Scale and Reshape for LSTM: (samples, 1, features)
    scaled_data = scaler.transform(processed_df)
    return scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])

def predict_electricity(new_data_df, model, scaler):
    """Makes predictions using the loaded model and scaler."""
    processed_data = preprocess_input(new_data_df, scaler)
    predictions = model.predict(processed_data)
    
    # Keras multi-output models return a list: [output_1, output_2]
    kwh_pred = predictions[0]
    bill_pred = predictions[1]
    
    return kwh_pred.flatten(), bill_pred.flatten()

# --- 3. STREAMLIT USER INTERFACE ---

st.title("⚡ Electricity Consumption & Cost Predictor")
st.markdown("Enter your home appliance details below to estimate your monthly energy and bill.")

# Load resources
multi_output_lstm_model, feature_scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

# Input Form
with st.form("prediction_form"):
    st.subheader("Appliance Details")
    col1, col2 = st.columns(2)
    
    with col1:
        ac_units = st.number_input("Number of Air Conditioners", min_value=0, value=1)
        ac_hp = st.number_input("AC Power (HP)", min_value=0.0, step=0.1, value=1.5)
        fridge = st.number_input("Refrigerators", min_value=0, value=1)
        tv = st.number_input("Televisions", min_value=0, value=2)
        fans = st.number_input("Fans", min_value=0, value=3)
        pc = st.number_input("Computers/Laptops", min_value=0, value=1)
        
    with col2:
        hours = st.slider("Average Daily Usage Hours", 0.0, 24.0, 8.0)
        house_size = st.number_input("House Size (m²)", min_value=10, value=100)
        washing = st.number_input("Washing Machine Usage (times/week)", 0, 20, 3)
        heater = st.selectbox("Has Water Heater?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        season = st.selectbox("Current Season", ["Summer", "Winter"])
        insulation = st.selectbox("Insulation Quality", ["High", "Medium", "Low"])

    submit_button = st.form_submit_button("Calculate Prediction")

# --- 4. PREDICTION LOGIC ---
if submit_button:
    # Build DataFrame from inputs
    input_df = pd.DataFrame([{
        'number_of_air_conditioners': ac_units,
        'ac_power_hp': ac_hp,
        'number_of_refrigerators': fridge,
        'number_of_televisions': tv,
        'number_of_fans': fans,
        'number_of_computers': pc,
        'average_daily_usage_hours': hours,
        'season': season.lower(),
        'house_size_m2': float(house_size),
        'insulation_quality': insulation.lower(),
        'has_water_heater': heater,
        'washing_machine_usage_per_week': washing
    }])

    with st.spinner("Calculating..."):
        try:
            kwh, bill = predict_electricity(input_df, multi_output_lstm_model, feature_scaler)
            
            st.success("Analysis Complete!")
            
            # Display metrics
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Predicted Consumption", f"{kwh[0]:.2f} kWh")
            res_col2.metric("Predicted Bill", f"{bill[0]:.2f} EGP")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.divider()
st.info("Note: This model uses an LSTM architecture for multi-output regression.")
