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
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #000000;
        color: white;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: white;
    }
    .stFileUploader {
        padding: 20px;
        border: 2px dashed #ccc;
        border-radius: 10px;
        text-align: center;
    }
    div.stSpinner > div {
        text-align:center;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Malaria Detection System")
st.markdown("""
<div style='text-align: center; color: #555; margin-bottom: 30px;'>
    Scientific Validation Tool based on Convolutional Neural Networks (CNN) & Grad-CAM Explainability.
    Upload a thin blood smear microscopy image to detect <i>Plasmodium</i> parasites.
</div>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('malaria_detection_model.h5', compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- GRAD-CAM EXPLAINABILITY FUNCTION (FIXED) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv_layer"):
    # Create a model that maps the input image to the activations of the last conv layer
    # and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        
        # ERROR FIX IS HERE:
        # Instead of complex slicing that caused the Tuple Error,
        # we directly access the index 0 since it is a binary output (1 neuron).
        class_channel = preds[:, 0]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_output)
    
    # Pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- MAIN INTERFACE (SINGLE PAGE LOGIC) ---

# 1. File Uploader Section
uploaded_file = st.file_uploader("Upload a microscopy image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.markdown("---")
    
    # 2. Layout: Image on Left, Analysis on Right (or Top/Down depending on mobile)
    # Using columns to center the workflow
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Sample")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis")
        st.write("Click the button below to start the AI inference process.")
        analyze_btn = st.button("RUN DIAGNOSIS")

    # 3. Execution Block
    if analyze_btn:
        with st.spinner('Processing image... Please wait.'):
            # Preprocessing
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            prediction = model.predict(img_array)[0][0]
            
            # Logic: < 0.5 is Parasitized (based on your training data), > 0.5 is Uninfected
            is_infected = prediction < 0.5 
            confidence = (1 - prediction) if is_infected else prediction
            
            # --- RESULTS SECTION ---
            st.markdown("### Results Report")
            
            # Create a clean container for results
            result_container = st.container()
            
            if is_infected:
                result_container.error("DIAGNOSIS: POSITIVE (INFECTED)")
                result_container.markdown(f"**Confidence Score:** {confidence*100:.2f}%")
                result_container.write("The AI model detected high probability of Plasmodium parasites.")
            else:
                result_container.success("DIAGNOSIS: NEGATIVE (HEALTHY)")
                result_container.markdown(f"**Confidence Score:** {confidence*100:.2f}%")
                result_container.write("No parasites detected. The cell appears healthy.")
            
            # --- GRAD-CAM VISUALIZATION ---
            st.markdown("---")
            st.subheader("Visual Evidence (Grad-CAM)")
            st.write("The red highlighted regions indicate where the AI detected the anomaly.")
            
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
                
                # Display Result
                st.image(superimposed, caption="AI Attention Map", use_container_width=True)
                
            except Exception as e:
                st.warning("Could not generate heatmap.")
                st.write(f"Technical details: {e}")
