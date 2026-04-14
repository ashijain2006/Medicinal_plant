
from flask import Flask, request, jsonify
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

app = Flask(__name__)

device = "cpu"

with open("merged_dataset/classes.txt") as f:
    class_names = [line.strip() for line in f]

model = timm.create_model(
    "vit_base_patch16_224_dino",
    pretrained=False,
    num_classes=len(class_names)
)

import gdown
import os

if not os.path.exists("model.pth"):
    url = "https://drive.google.com/uc?id=1K2GDj9DV-Cz6mUMSMf-1m2Pq_i0_2y0_"
    gdown.download(url, "model.pth", quiet=False, fuzzy=True, use_cookies=False)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

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

    pred = torch.argmax(output).item()

    return jsonify({"prediction": class_names[pred]})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
