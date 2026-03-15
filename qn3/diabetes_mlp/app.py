import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Custom CSS for Light Theme
st.set_page_config(
    page_title="Diabetes Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light theme styling
custom_css = """
    <style>
        :root {
            --primary-color: #EC4899;
            --secondary-color: #6366F1;
            --success-color: #10B981;
            --danger-color: #EF4444;
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
            background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%);
            color: white;
            padding: 20px 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(236, 72, 153, 0.15);
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
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .result-positive {
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        }
        
        .result-negative {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        }
        
        .result-text {
            font-size: 1.4em;
            font-weight: 700;
            margin: 10px 0;
            letter-spacing: 0.5px;
        }
        
        .result-label {
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%);
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
            box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);
            transform: translateY(-2px);
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
            margin-bottom: 3px !important;
        }
        
        h3 {
            margin-top: 8px !important;
            margin-bottom: 10px !important;
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
        <h1>🩺 Diabetes Prediction</h1>
        <p>Medical Risk Assessment System</p>
    </div>
""", unsafe_allow_html=True)

# Main Content
st.markdown("<h3 style='color: #000000;'>📋 Patient Medical Details</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-label'>Pregnancies</div>", unsafe_allow_html=True)
    Pregnancies = st.number_input("Pregnancies", 0, 20, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Glucose</div>", unsafe_allow_html=True)
    Glucose = st.number_input("Glucose", 0, 200, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Blood Pressure</div>", unsafe_allow_html=True)
    BloodPressure = st.number_input("Blood Pressure", 0, 150, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Skin Thickness</div>", unsafe_allow_html=True)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, label_visibility="collapsed")

with col2:
    st.markdown("<div class='input-label'>Insulin</div>", unsafe_allow_html=True)
    Insulin = st.number_input("Insulin", 0, 900, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>BMI</div>", unsafe_allow_html=True)
    BMI = st.number_input("BMI", 0.0, 70.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Diabetes Pedigree Function</div>", unsafe_allow_html=True)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, label_visibility="collapsed")
    
    st.markdown("<div class='input-label'>Age</div>", unsafe_allow_html=True)
    Age = st.number_input("Age", 1, 120, label_visibility="collapsed")

st.markdown("---")

# Predict Button
if st.button("🔍 Analyze Patient"):
    
    input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                            SkinThickness, Insulin, BMI,
                            DiabetesPedigreeFunction, Age]])
    
    input_scaled = scaler.transform(input_data)
    
    with st.spinner("🤖 Analyzing patient data..."):
        prediction = model.predict(input_scaled, verbose=0)
    
    # Results Section
    if prediction[0][0] > 0.5:
        st.markdown(f"""
            <div class='result-card result-positive'>
                <div class='result-label'>PREDICTION RESULT</div>
                <div class='result-text'>⚠️ Diabetic Risk Detected</div>
                <div style='font-size: 0.9em; opacity: 0.95;'>Risk Score: {prediction[0][0]*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='result-card result-negative'>
                <div class='result-label'>PREDICTION RESULT</div>
                <div class='result-text'>✓ Low Diabetes Risk</div>
                <div style='font-size: 0.9em; opacity: 0.95;'>Risk Score: {prediction[0][0]*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)