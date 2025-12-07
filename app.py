import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Malaria Diagnosis AI", layout="centered")

# --- CSS STYLING (Clean & Professional) ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #333333;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #000000;
        color: white;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
    }
    .stFileUploader {
        padding: 20px;
        border: 1px dashed #ccc;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Malaria Detection System")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Scientific Validation Tool based on Convolutional Neural Networks (CNN) & Grad-CAM Explainability.
    Upload a thin blood smear microscopy image to detect <i>Plasmodium</i> parasites.
</div>
<hr>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('malaria_detection_model.h5', compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- GRAD-CAM EXPLAINABILITY FUNCTION ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv_layer"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        # BINARY CLASSIFICATION FIX:
        # We take the 0th index directly since there is only one output neuron
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- MAIN INTERFACE (SINGLE PAGE) ---

# File Uploader in the main body
uploaded_file = st.file_uploader("Choose a microscopy image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Sample", use_container_width=True)
        
        # Analyze Button
        analyze_btn = st.button("ANALYZE SAMPLE")

    if analyze_btn:
        with st.spinner('Processing image with AI model...'):
            # Preprocessing
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            prediction = model.predict(img_array)[0][0]
            
            # Logic: < 0.5 is Parasitized (based on training), > 0.5 is Uninfected
            is_infected = prediction < 0.5 
            confidence = (1 - prediction) if is_infected else prediction
            
            # Separator
            st.markdown("---")
            
            # Results Layout
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Diagnostic Result")
                if is_infected:
                    st.error("POSITIVE (INFECTED)")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    st.write("The model detected the presence of parasites.")
                else:
                    st.success("NEGATIVE (HEALTHY)")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    st.write("No parasites were detected in this sample.")
            
            with res_col2:
                st.subheader("Visual Evidence (Grad-CAM)")
                try:
                    # Generate Heatmap
                    heatmap = make_gradcam_heatmap(img_array, model)
                    
                    # Resize and Colorize
                    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Superimpose
                    original_img = np.array(image)
                    superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
                    
                    st.image(superimposed, caption="Red areas indicate parasite location", use_container_width=True)
                except Exception as e:
                    st.warning("Could not generate heatmap.")
                    st.write(e)
