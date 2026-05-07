from fastapi import FastAPI,UploadFile,File
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import io


app = FastAPI(
    title="Detection DeepFake",
    description="API pour détecter les images fake avec ResNet50",
    version="1.0"
)

device = torch.device("cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(
    torch.load("best_model_ft.pth", map_location=device)
)

model.eval()
print("Modèle est chargé")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
])

@app.get("/")
def home():
    return {
        "message":"API Deepfake detection",
        "status":"running",
        "model":"RestNet50 fine-tuned",
        "accuracy":"98.86%"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output,dim=1)
        pred_class = output.argmax(dim=1).item()

        label = "FAKE" if pred_class == 0 else "REAL"
        confidence = probabilities[0][pred_class].item() *100

        return {
            "prediction": label,
            "confidence": round(confidence,2),
            "score_fake": round(probabilities[0][0].item()*100,2),
            "score_real": round(probabilities[0][1].item()*100,2)
        }
