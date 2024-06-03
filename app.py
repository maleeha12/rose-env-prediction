import streamlit as st
import joblib
import numpy as np

# Load the models
random_forest_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title('Environment Prediction')

# Create the form
HS_analog = st.number_input('HS Analog:', format="%f")
L_lux = st.number_input('L Lux:', format="%f")
T_deg = st.number_input('Temperature (°C):', format="%f")
CO2_analog = st.number_input('CO2 Analog:', format="%f")
HR_percent = st.number_input('Humidity (%):', format="%f")

if st.button('Predict'):
    try:
        # Prepare the feature array
        features = np.array([[HS_analog, L_lux, T_deg, CO2_analog, HR_percent]])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make the prediction
        prediction = random_forest_model.predict(scaled_features)

        # Interpret the prediction
        if prediction == 1:
            result = "Dry soil"
        elif prediction == 2:
            result = "Good environment"
        elif prediction == 3:
            result = "Too hot"
        elif prediction == 4:
            result = "Too cold environment"
        else:
            result = "Unknown"
        
        st.success(f'Result: {result}')
    except Exception as e:
        st.error(f"Error: {e}")

# Styling with Markdown
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #e0f7fa;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        padding: 0 20px;
    }
    .container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 400px;
    }
    h1 {
        color: #00796b;
        text-align: center;
        font-size: 1.8em;
    }
    label {
        margin-top: 15px;
        color: #004d40;
        font-size: 1.1em;
    }
    .stNumberInput > div {
        padding: 10px;
        margin-top: 5px;
        border: 1px solid #b2dfdb;
        border-radius: 5px;
        font-size: 1em;
    }
    button {
        margin-top: 20px;
        padding: 10px;
        background-color: #00796b;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    button:hover {
        background-color: #004d40;
    }
    .result {
        margin-top: 20px;
        padding: 10px;
        text-align: center;
        background-color: #e0f2f1;
        border: 1px solid #b2dfdb;
        border-radius: 5px;
        font-size: 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)
