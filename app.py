import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

# --- 1. SETUP & CACHED LOAD ---
st.set_page_config(page_title="AI Pneumonia Detector", page_icon="🩻", layout="wide")

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('xray_model.keras')
    return model

model = load_model()

# --- 2. GRAD-CAM FUNCTION (The Robust Fix) ---
def make_gradcam_heatmap(img_array, model):
    # A. Find the last convolutional layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name:
            last_conv_layer_name = layer.name
            break
    
    # B. Explicit Model Reconstruction (Fixes the Keras 3 'never been called' error)
    new_input = tf.keras.Input(shape=(64, 64, 3))
    x = new_input
    last_conv_output = None
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    # C. Create the Grad-CAM model with explicit inputs/outputs
    grad_model = tf.keras.models.Model(
        inputs=new_input,
        outputs=[last_conv_output, x]
    )

    # D. Compute Gradients (The rest of the Grad-CAM math)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 3. UI LAYOUT ---
st.title("🩻 AI Radiologist: Explainable Diagnostics")
st.markdown("Upload a Chest X-Ray. The AI will detect Pneumonia and **highlight the infected area**.")

col1, col2 = st.columns(2)

# --- 4. PREDICTION LOGIC ---
uploaded_file = st.sidebar.file_uploader("Choose an X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and Preprocess
    img_pil = Image.open(uploaded_file).convert('RGB')
    img_array = image.img_to_array(img_pil.resize((64, 64)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    # Display Original Image in Column 1
    with col1:
        st.subheader("Original X-Ray")
        st.image(img_pil, use_column_width=True)

    # Display Result
    if probability > 0.5:
        st.sidebar.error(f"🚨 **PNEUMONIA DETECTED**")
        st.sidebar.write(f"Confidence: {probability:.2%}")
        
        # --- 5. GENERATE HEATMAP VISUALIZATION ---
        with col2:
            st.subheader("AI Analysis (Heatmap)")
            with st.spinner("Generating explanation..."):
                # 1. Get the raw heatmap
                heatmap = make_gradcam_heatmap(img_array, model)
                
                # 2. Visualization Setup
                img_cv2 = np.array(img_pil)
                img_cv2 = img_cv2[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV
                
                # 3. Superimpose
                heatmap = cv2.resize(heatmap, (img_cv2.shape[1], img_cv2.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                superimposed_img = heatmap * 0.4 + img_cv2 * 0.6
                superimposed_img = np.uint8(superimposed_img)
                
                # 4. Show it!
                st.image(superimposed_img, caption="Red = Infected Area", channels="BGR", use_column_width=True)
                
    else:
        st.sidebar.success(f"✅ **NORMAL (Healthy)**")
        st.sidebar.write(f"Confidence: {(1-probability):.2%}")