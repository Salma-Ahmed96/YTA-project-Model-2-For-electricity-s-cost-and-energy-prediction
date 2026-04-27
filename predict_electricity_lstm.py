
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import joblib
import os

def load_model_and_scaler(model_path, scaler_path):
    """Loads the Keras model and the StandardScaler."""
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_input(new_data_df, scaler):
    """Preprocesses new input data for the LSTM model."""
    # Ensure columns match the training features (X_train.columns)
    # These columns should be consistent with what was used for training.
    # For this specific dataset, we know the original categorical columns.

    # Identify categorical columns that were one-hot encoded during training
    categorical_cols = ['season', 'insulation_quality']
    
    # Make a copy to avoid modifying the original DataFrame
    processed_df = new_data_df.copy()

    # One-hot encode new categorical features (if they exist)
    # Ensure all original categories are represented, even if not in new_data_df
    # This assumes 'summer' and 'high' were the reference categories (dropped first)
    # and that 'season_winter', 'insulation_quality_low', 'insulation_quality_medium' 
    # are the resulting dummy variables.

    # Example of handling 'season'
    if 'season' in processed_df.columns:
        processed_df['season_winter'] = (processed_df['season'] == 'winter').astype(int)
        processed_df = processed_df.drop(columns=['season'])
    else:
        # If 'season' column is missing, assume default (e.g., all summer or handle as appropriate)
        processed_df['season_winter'] = 0 # Default to summer if not provided

    # Example of handling 'insulation_quality'
    if 'insulation_quality' in processed_df.columns:
        processed_df['insulation_quality_low'] = (processed_df['insulation_quality'] == 'low').astype(int)
        processed_df['insulation_quality_medium'] = (processed_df['insulation_quality'] == 'medium').astype(int)
        processed_df = processed_df.drop(columns=['insulation_quality'])
    else:
        # If 'insulation_quality' column is missing, assume default (e.g., all high or handle as appropriate)
        processed_df['insulation_quality_low'] = 0
        processed_df['insulation_quality_medium'] = 0

    # Ensure the order and presence of all feature columns (excluding target columns)
    # based on X_train.columns from your training phase.
    # Replace with the actual column names from your X_train after dropping targets.
    expected_feature_columns = ['number_of_air_conditioners', 'ac_power_hp', 'number_of_refrigerators',
                                'number_of_televisions', 'number_of_fans', 'number_of_computers',
                                'average_daily_usage_hours', 'house_size_m2', 'has_water_heater',
                                'washing_machine_usage_per_week', 'monthly_energy_consumption_kwh',
                                'season_winter', 'insulation_quality_low', 'insulation_quality_medium']
    
    # Filter and reorder columns to match the training data
    # Note: 'monthly_energy_consumption_kwh' should NOT be included here if it's an output, 
    # but it was included in X for the LSTM model. If it's a feature for the LSTM, 
    # make sure it's present in input data or imputed.
    
    # For this specific model, 'monthly_energy_consumption_kwh' was one of the two outputs, 
    # not an input feature. It needs to be removed from the expected_feature_columns for the input X.
    # Correcting expected_feature_columns based on X from f4e51433, which is `df.drop(['monthly_electricity_bill_egp', 'monthly_energy_consumption_kwh'], axis=1)`
    expected_feature_columns = ['number_of_air_conditioners', 'ac_power_hp', 'number_of_refrigerators',
                                'number_of_televisions', 'number_of_fans', 'number_of_computers',
                                'average_daily_usage_hours', 'house_size_m2', 'has_water_heater',
                                'washing_machine_usage_per_week', 'season_winter', 
                                'insulation_quality_low', 'insulation_quality_medium']

    # Add any missing columns with default values (e.g., 0) if they were features during training
    for col in expected_feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 # Or a suitable default/imputation strategy

    # Ensure all expected columns are present and in the correct order
    processed_df = processed_df[expected_feature_columns]

    # Scale numerical features
    scaled_data = scaler.transform(processed_df)
    
    # Reshape for LSTM input: (samples, timesteps, features)
    # Assuming 1 timestep per feature set
    reshaped_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
    
    return reshaped_data

def predict_electricity(new_data_df, model, scaler):
    """Makes predictions using the loaded model and scaler."""
    processed_data = preprocess_input(new_data_df, scaler)
    kwh_pred, bill_pred = model.predict(processed_data)
    return kwh_pred.flatten(), bill_pred.flatten()

if __name__ == '__main__':
    # Example usage (for Streamlit, this would be integrated into the app logic)
    MODEL_PATH = "trained_models/multi_output_lstm_model.h5"
    SCALER_PATH = "trained_models/feature_scaler.joblib"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler not found at {SCALER_PATH}")

    # Load the model and scaler
    multi_output_lstm_model, feature_scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

    print("Model and scaler loaded successfully.")

    # Create some example new data (replace with actual Streamlit input)
    example_new_data = pd.DataFrame([
        {
            'number_of_air_conditioners': 2,
            'ac_power_hp': 1.5,
            'number_of_refrigerators': 1,
            'number_of_televisions': 3,
            'number_of_fans': 2,
            'number_of_computers': 1,
            'average_daily_usage_hours': 8.5,
            'season': 'summer', # Categorical input
            'house_size_m2': 120.0,
            'insulation_quality': 'medium', # Categorical input
            'has_water_heater': 1,
            'washing_machine_usage_per_week': 3
        },
        {
            'number_of_air_conditioners': 0,
            'ac_power_hp': 0.0, # Not applicable if AC units is 0
            'number_of_refrigerators': 2,
            'number_of_televisions': 1,
            'number_of_fans': 4,
            'number_of_computers': 0,
            'average_daily_usage_hours': 5.0,
            'season': 'winter', # Categorical input
            'house_size_m2': 80.0,
            'insulation_quality': 'high', # Categorical input
            'has_water_heater': 0,
            'washing_machine_usage_per_week': 1
        }
    ])

    # Make predictions
    predicted_kwh, predicted_bill = predict_electricity(example_new_data, multi_output_lstm_model, feature_scaler)

    print("--- Example Predictions ---")
    for i in range(len(example_new_data)):
        print(f"Input {i+1}:")
        print(example_new_data.iloc[i])
        print(f"  Predicted Monthly Energy Consumption (kWh): {predicted_kwh[i]:.2f}")
        print(f"  Predicted Monthly Electricity Bill (EGP): {predicted_bill[i]:.2f}")
        print("------------------------")

