import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- 1. SETUP & LOAD MODEL ---
st.set_page_config(page_title="AI Pneumonia Detector", page_icon="🩻")

@st.cache_resource
def load_model():
    # Load the trained CNN model
    model = tf.keras.models.load_model('xray_model.keras')
    return model

model = load_model()

st.title("🩻 AI Radiologist: Pneumonia Detector")
st.markdown("Upload a Chest X-Ray image to detect if the patient has Pneumonia.")

# --- 2. IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose an X-Ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    # FIX: Force convert to 'RGB' to ensure it has 3 channels (matches the model)
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded X-Ray', use_column_width=True)
    
    st.write("scanning image...")

    # --- 3. PREPROCESSING ---
    img = img.resize((64, 64))
    
    # Convert to array and add batch dimension (1, 64, 64, 3)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the pixel values (just like we did in training: 1./255)
    img_array = img_array / 255.0

    # --- 4. PREDICTION ---
    prediction = model.predict(img_array)
    
    # The output is a probability (0 to 1)
    # Based on our training: 0 = Normal, 1 = Pneumonia
    probability = prediction[0][0]

    st.write("---")
    st.subheader("Diagnosis Result:")
    
    if probability > 0.5:
        # High probability of Pneumonia
        confidence = probability * 100
        st.error(f"🚨 **PNEUMONIA DETECTED**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.warning("⚠️ Recommendation: Consult a Pulmonologist immediately.")
    else:
        # Low probability (Normal)
        confidence = (1 - probability) * 100
        st.success(f"✅ **NORMAL (Healthy)**")
        st.write(f"Confidence: **{confidence:.2f}%**")