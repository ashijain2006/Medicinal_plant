from flask import Flask, request, jsonify, render_template
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import gdown
import pandas as pd
import numpy as np
import cv2

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

if not os.path.exists("model.pth"):
    url = "https://drive.google.com/uc?id=1K2GDj9DV-Cz6mUMSMf-1m2Pq_i0_2y0_"
    gdown.download(url, "model.pth", quiet=False)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# XAI HEATMAP FUNCTION
# =========================
def generate_heatmap(model, img_tensor):
    img_tensor.requires_grad = True

    output = model(img_tensor)
    pred_class = output.argmax()

    output[0, pred_class].backward()

    gradients = img_tensor.grad.data.numpy()[0]
    heatmap = np.mean(gradients, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    return heatmap

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================
# UI ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]

        img = Image.open(file).convert("RGB")
        img_resized = img.resize((224,224))

        img_tensor = transform(img).unsqueeze(0).to(device)
        img_tensor.requires_grad = True

        # 🔥 Forward pass (NO torch.no_grad)
        output = model(img_tensor)

        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

        plant_name = class_names[pred]

        # 🔥 XAI
        heatmap = generate_heatmap(model, img_tensor)

        img_np = np.array(img_resized)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        os.makedirs("static", exist_ok=True)
        cv2.imwrite("static/heatmap.jpg", overlay)

        # 📖 INFO
        row = df[df["name"] == plant_name]

        if not row.empty:
            uses = row.iloc[0]["uses"]
            desc = row.iloc[0]["description"]
            info = f"Uses: {uses}\n\nDescription: {desc}"
        else:
            info = "No data available"

        return render_template(
            "index.html",
            prediction=plant_name,
            confidence=f"{confidence*100:.2f}%",
            info=info,
            image_path="static/heatmap.jpg"
        )

    return render_template("index.html")

# =========================
# API ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = Image.open(file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # API can use no_grad (faster)
    with torch.no_grad():
        output = model(img_tensor)

    probs = torch.softmax(output, dim=1)[0]
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    plant_name = class_names[pred]

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
