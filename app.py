import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Sayfa Ayarları
st.set_page_config(page_title="AI Malaria Diagnosis", layout="wide")

# Google Sites için Temiz Görünüm CSS
st.markdown("""
<style>
    .main { background-color: white; }
    h1 { color: #d32f2f; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #d32f2f; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🩸 Sıtma Teşhis Yapay Zekası")
st.markdown("**Bilimsel Yöntem:** Convolutional Neural Networks (CNN) & Grad-CAM")

# Modeli Önbellekle (Hız için)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('malaria_detection_model.h5', compile=False)

model = load_model()

# Kenar Çubuğu
st.sidebar.header("🔬 Görüntü Yükleme")
uploaded_file = st.sidebar.file_uploader("Mikroskop Görüntüsü Yükle", type=["jpg", "png", "jpeg"])

# Grad-CAM Fonksiyonu
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv_layer"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Örnek", width=300)
    
    if st.button("Analiz Et"):
        with st.spinner('Yapay Zeka hücreyi inceliyor...'):
            # Ön İşleme
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Tahmin
            prediction = model.predict(img_array)[0][0]
            # Veri setine göre: 0=Parazitli, 1=Sağlıklı (Bazen tam tersi olabilir, etiketi kontrol ediyoruz)
            # Eğitimde 0=Parasitized ise:
            is_infected = prediction < 0.5 
            confidence = (1 - prediction) if is_infected else prediction
            
            # Sonuç Gösterimi
            col1, col2 = st.columns(2)
            
            with col1:
                if is_infected:
                    st.error(f"### SONUÇ: POZİTİF (ENFEKTE)")
                    st.write(f"**Güven Oranı:** %{confidence*100:.2f}")
                else:
                    st.success(f"### SONUÇ: NEGATİF (TEMİZ)")
                    st.write(f"**Güven Oranı:** %{confidence*100:.2f}")

            with col2:
                # Isı Haritası
                heatmap = make_gradcam_heatmap(img_array, model)
                heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                superimposed = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
                st.image(superimposed, caption="AI Dikkat Haritası (Kırmızı alanlar paraziti gösterir)")