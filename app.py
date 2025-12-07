import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Malaria Diagnosis AI", layout="centered")

# --- CSS STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #333333;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #000000;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        border: none;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: white;
    }
    .stFileUploader {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 20px;
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
<div style='text-align: center; color: #666; font-size: 16px; margin-bottom: 30px;'>
    Scientific Validation Tool based on CNN & Grad-CAM Explainability.
    Upload a blood smear image to detect <i>Plasmodium</i> parasites.
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

# --- GRAD-CAM EXPLAINABILITY FUNCTION (ROBUST FIX) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv_layer"):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        
        # --- CRITICAL FIX IS HERE ---
        # Instead of 'preds[:, 0]' which fails if preds is a list,
        # we use 'preds[0]' which works for both Lists and Tensors.
        # This grabs the prediction for the single image in the batch.
        class_channel = preds[0]

    # Compute gradients of the class channel with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_output)
    
    # Pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- MAIN LOGIC ---

uploaded_file = st.file_uploader("Upload a microscopy image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.markdown("---")
    
    # Layout using Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Sample")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis")
        st.write("Click below to run the diagnostic AI model.")
        analyze_btn = st.button("RUN DIAGNOSIS")

    # Execution Block
    if analyze_btn:
        with st.spinner('AI is processing the image...'):
            # 1. Preprocessing
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # 2. Prediction
            prediction = model.predict(img_array)[0][0]
            
            # Logic: Training labels were 0=Parasitized, 1=Uninfected
            is_infected = prediction < 0.5 
            confidence = (1 - prediction) if is_infected else prediction
            
            # 3. Results Container
            st.markdown("### Results Report")
            result_container = st.container()
            
            if is_infected:
                result_container.error("DIAGNOSIS: POSITIVE (INFECTED)")
                result_container.markdown(f"**Confidence Score:** {confidence*100:.2f}%")
                result_container.write("The AI detected Plasmodium parasites in this cell.")
            else:
                result_container.success("DIAGNOSIS: NEGATIVE (HEALTHY)")
                result_container.markdown(f"**Confidence Score:** {confidence*100:.2f}%")
                result_container.write("No parasites detected. The cell is healthy.")
            
            # 4. Grad-CAM Visualization
            st.markdown("---")
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
                
                # Display
                st.image(superimposed, caption="Red areas indicate AI attention (Parasite location)", use_container_width=True)
                
            except Exception as e:
                st.warning("Could not generate heatmap.")
                st.code(f"Error: {e}")
