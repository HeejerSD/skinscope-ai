import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="SkinScope AI",
    page_icon="🩺",
    layout="centered"
)

# -------- CUSTOM CSS STYLE --------
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe5e5;
    }

   h1 {
    color: #6d0019;  /* bordeaux foncé premium */
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}

h3 {
    color: #8b0030;  /* bordeaux plus lumineux */
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

    .confidence-box {
        background-color: #fff0f0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        color: #800020;
        margin-top: 10px;
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

# -------- CONFIG --------
num_classes = 7  # DermaMNIST

class_names = [
    "Actinic keratoses and intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions"
]

# -------- LOAD MODEL --------
model = models.resnet18(pretrained=False)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

model.load_state_dict(torch.load("best_resnet_tl.pth", map_location="cpu"))
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# -------- FILE UPLOADER --------
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    # -------- DISPLAY RESULT --------
    st.markdown(f"""
        <div class="prediction-box">
            Prediction:<br>{predicted_class}
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="confidence-box">
            Confidence: {confidence_score:.2f}%
        </div>
    """, unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("""
    <div class="footer">
        Developed with Deep Learning • ResNet18 Transfer Learning • DermaMNIST Dataset
    </div>
""", unsafe_allow_html=True)