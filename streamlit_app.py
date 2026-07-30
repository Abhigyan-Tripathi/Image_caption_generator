# forcing rebuild
import streamlit as st
from PIL import Image
import numpy as np
import os
import urllib.request
from utils import load_resources, generate_caption

# --- 1. PASTE YOUR HF HUB URLS HERE ---
# (See Step 2 on how to get these links)
MODEL_URL = "https://huggingface.co/Abhigyan05/Image-caption-resnet-transformer/resolve/main/image_captioner.keras"
VOCAB_URL = "https://huggingface.co/Abhigyan05/Image-caption-resnet-transformer/resolve/main/vocab.json"

# --- 2. Download & Load Model (Cached so it only runs once per server boot) ---
@st.cache_resource
def load_model_and_vocab():
    if not os.path.exists("image_captioner.keras"):
        with st.spinner("Downloading model weights... (This happens only on cold start)"):
            urllib.request.urlretrieve(MODEL_URL, "image_captioner.keras")
    if not os.path.exists("vocab.json"):
        urllib.request.urlretrieve(VOCAB_URL, "vocab.json")
        
    return load_resources("image_captioner.keras", "vocab.json")

model, stoi, itos = load_model_and_vocab()

# --- 3. Build the UI ---
st.set_page_config(page_title="Image Captioner", layout="centered")

st.title("🖼️ Image Caption Generator")
st.markdown("Powered by a **ResNet50 Encoder** and a **Custom Transformer Decoder** built from scratch.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Caption 🪄"):
        with st.spinner("Analyzing image patches with Cross-Attention..."):
            img_array = np.array(image)
            caption = generate_caption(model, stoi, itos, img_array)
            
        st.success(f"**Generated Caption:** {caption}")
