from flask import Flask, request, jsonify
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import gdown
import pandas as pd

app = Flask(__name__)

device = "cpu"

# =========================
# LOAD CLASS NAMES
# =========================
with open("merged_dataset/classes.txt") as f:
    class_names = [line.strip() for line in f]

# =========================
# LOAD PLANT INFO CSV
# =========================
df = pd.read_csv("plant_info.csv")

# =========================
# LOAD MODEL
# =========================
model = timm.create_model(
    "vit_base_patch16_224_dino",
    pretrained=False,
    num_classes=len(class_names)
)

# Download model if not exists
if not os.path.exists("model.pth"):
    url = "https://drive.google.com/uc?id=1K2GDj9DV-Cz6mUMSMf-1m2Pq_i0_2y0_"
    gdown.download(url, "model.pth", quiet=False)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    probs = torch.softmax(output, dim=1)[0]
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    plant_name = class_names[pred]

    # =========================
    # KNOWLEDGE MAPPING
    # =========================
    row = df[df["name"] == plant_name]

    if not row.empty:
        uses = row.iloc[0]["uses"]
        desc = row.iloc[0]["description"]
        info = f"Uses: {uses}\n\nDescription: {desc}"
    else:
        info = "No data available"

    return jsonify({
        "prediction": plant_name,
        "confidence": f"{confidence*100:.2f}%",
        "info": info
    })

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
