import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import gdown

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="SkinScope AI",
    page_icon="🩺",
    layout="centered"
)

# -------- STYLE --------
st.markdown("""
<style>
.stApp { background-color: #ffe5e5; }

h1 {
    color: #6d0019;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}

h3 {
    color: #8b0030;
    text-align: center;
    font-weight: 600;
}

.prediction-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #800020;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-top: 20px;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #800020;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1>SkinScope AI</h1>", unsafe_allow_html=True)
st.markdown("<h3>Intelligent Dermoscopic Analysis System</h3>", unsafe_allow_html=True)
st.write("---")

# -------- MODEL CONFIG --------
MODEL_PATH = "best_resnet_tl.pth"
FILE_ID = "1OlSbUziq66wFF7dznV4ljigExgZ28x9b"

@st.cache_resource
def load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... please wait.")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False,
            fuzzy=True  # IMPORTANT for large files
        )

    # Check download
    if not os.path.exists(MODEL_PATH):
        st.error("Model download failed. Check Google Drive permissions.")
        st.stop()

    # Build model
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 7)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------- CLASS NAMES --------
class_names = [
    "Actinic keratoses and intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions"
]

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# -------- FILE UPLOADER --------
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    st.markdown(f"""
    <div class="prediction-box">
        Prediction:<br>{predicted_class}
        <br><br>
        Confidence: {confidence_score:.2f}%
    </div>
    """, unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("""
<div class="footer">
Developed with Deep Learning • ResNet18 Transfer Learning • DermaMNIST Dataset
</div>
""", unsafe_allow_html=True)