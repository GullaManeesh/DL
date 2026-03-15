import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model
model = load_model("model.h5", compile=False)

# Custom CSS for Light Theme
st.set_page_config(
    page_title="Digit Recognition AI",
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
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            color: white;
            padding: 20px 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15);
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
        
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            border: 2px dashed #E5E7EB;
            margin-bottom: 20px;
        }
        
        .result-card {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        }
        
        .confidence-bar {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            height: 6px;
            margin: 5px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: white;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        .prediction-number {
            font-size: 2.5em;
            font-weight: 700;
            margin: 8px 0;
        }
        
        .info-card {
            background: #EEF2FF;
            border-left: 4px solid #6366F1;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
        
        /* File uploader styling */
        [data-testid="stFileUploadDropzone"] {
            background: white !important;
            border: 2px dashed #6366F1 !important;
            border-radius: 12px !important;
            color: #1F2937 !important;
        }
        
        [data-testid="stFileUploadDropzone"] p {
            color: #1F2937 !important;
            font-weight: 600;
        }
        
        /* Hide file uploader info message */
        [data-testid="uploadedFileName"] {
            display: none !important;
        }
        
        .stFileUploader {
            margin: 10px 0 !important;
        }
        
        /* Compact spacing */
        .stMarkdown {
            margin-bottom: 5px !important;
        }
        
        h3 {
            margin-top: 8px !important;
            margin-bottom: 8px !important;
        }
        
        h4 {
            margin-top: 8px !important;
            margin-bottom: 8px !important;
        }
        
        hr {
            margin: 8px 0 !important;
        }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Main Header
st.markdown("""
    <div class='main-header'>
        <h1>🔢 Smart Digit Recognition</h1>
        <p>AI-Powered Handwritten Digit Classifier</p>
    </div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("<h3 style='color: #000000;'>📤 Upload Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    st.markdown("---")
    
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Ensure size is 28x28
    img = cv2.resize(img, (28, 28))
    
    # Display image in col1
    with col1:
        st.image(img, caption="Uploaded Image", width=150)
    
    # Normalize pixel values
    img_norm = img / 255.0
    
    # Reshape for model
    img_input = img_norm.reshape(1, 28, 28)
    
    # Predict
    with st.spinner("🤖 Analyzing image..."):
        prediction = model.predict(img_input, verbose=0)
        digit = np.argmax(prediction)
        confidence = float(prediction[0][digit]) * 100
    
    with col2:
        st.markdown("<h3 style='color: #000000;'>🎯 Prediction Result</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='result-card'>
                <div style='font-size: 0.9em; opacity: 0.9;'>PREDICTED DIGIT</div>
                <div class='prediction-number'>{digit}</div>
                <div style='font-size: 1.1em; font-weight: 600;'>Confidence: {confidence:.2f}%</div>
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence}%'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)