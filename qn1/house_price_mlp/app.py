import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Custom CSS for Light Theme
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light theme styling
custom_css = """
    <style>
        :root {
            --primary-color: #6366F1;
            --secondary-color: #EC4899;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --light-bg: #F9FAFB;
            --card-bg: #FFFFFF;
            --text-primary: #1F2937;
            --text-secondary: #6B7280;
            --border-color: #E5E7EB;
        }
        
        body {
            background-color: #F3F4F6;
        }
        
        .stApp {
            background-color: #F3F4F6;
        }
        
        .main-header {
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            color: white;
            padding: 20px 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.15);
            text-align: center;
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 1.8em;
            font-weight: 700;
        }
        
        .main-header p {
            margin: 5px 0 0 0;
            font-size: 0.85em;
            opacity: 0.95;
        }
        
        .result-card {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
            text-align: center;
        }
        
        .price-display {
            font-size: 2.5em;
            font-weight: 700;
            margin: 10px 0;
            letter-spacing: 1px;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 32px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            font-size: 14px;
        }
        
        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
            transform: translateY(-2px);
        }
        
        .input-section {
            background: white;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            margin-bottom: 10px;
        }
        
        .input-label {
            color: #000000;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 4px;
        }
        
        /* Compact spacing */
        .stNumberInput {
            margin-bottom: 5px !important;
        }
        
        .stMarkdown {
            margin-bottom: 2px !important;
        }
        
        h3 {
            margin-top: 10px !important;
            margin-bottom: 12px !important;
        }
        
        hr {
            margin: 10px 0 !important;
        }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Main Header
st.markdown("""
    <div class='main-header'>
        <h1>🏠 House Price Prediction</h1>
        <p>AI-Powered Real Estate Valuation System</p>
    </div>
""", unsafe_allow_html=True)

# Main Content
st.markdown("<h3 style='color: #000000;'>📝 Enter Property Details</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-label'>Median Income</div>", unsafe_allow_html=True)
    MedInc = st.number_input("Median Income", value=5.0, min_value=0.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>House Age</div>", unsafe_allow_html=True)
    HouseAge = st.number_input("House Age", value=20.0, min_value=0.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Average Rooms</div>", unsafe_allow_html=True)
    AveRooms = st.number_input("Average Rooms", value=6.0, min_value=0.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Average Bedrooms</div>", unsafe_allow_html=True)
    AveBedrms = st.number_input("Average Bedrooms", value=1.0, min_value=0.0, label_visibility="collapsed")

with col2:
    st.markdown("<div class='input-label'>Population</div>", unsafe_allow_html=True)
    Population = st.number_input("Population", value=900.0, min_value=0.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Average Occupancy</div>", unsafe_allow_html=True)
    AveOccup = st.number_input("Average Occupancy", value=3.0, min_value=0.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Latitude</div>", unsafe_allow_html=True)
    Latitude = st.number_input("Latitude", value=34.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Longitude</div>", unsafe_allow_html=True)
    Longitude = st.number_input("Longitude", value=-118.0, label_visibility="collapsed")

st.markdown("---")

# Predict Button
if st.button("💰 Predict Price"):
    
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
    
    input_scaled = scaler.transform(input_data)
    
    with st.spinner("🤖 Calculating price..."):
        prediction = model.predict(input_scaled, verbose=0)
        price = prediction[0][0] * 100000
    
    # Results Section
    st.markdown("""
        <div class='result-card'>
            <div style='font-size: 1em; opacity: 0.9;'>ESTIMATED PROPERTY VALUE</div>
            <div class='price-display'>${:,.2f}</div>
        </div>
    """.format(price), unsafe_allow_html=True)