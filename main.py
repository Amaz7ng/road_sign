from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from model import TrafficSignCNN

app = FastAPI(
    title="Распознавание дорожных знаков",
    description="Загрузи фото знака, и нейросеть скажет, что это!"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TrafficSignCNN()
model.load_state_dict(torch.load("traffic_sign_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

SIGN_NAMES = {
    0: "Ограничение скорости 20 км/ч",
    1: "Ограничение скорости 30 км/ч",
    2: "Ограничение скорости 50 км/ч",
    3: "Ограничение скорости 60 км/ч",
    14: "Дорожный знак: СТОП (Stop)",
    17: "Дорожный знак: Въезд запрещен (Кирпич)",
    21: "Опасный поворот направо",
    28: "Осторожно, дети!"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(input_tensor)
            _, predicted_class = torch.max(logits, 1)
            class_id = predicted_class.item()
            
        sign_text = SIGN_NAMES.get(class_id, f"Знак определен (Внутренний класс модели: {class_id})")
        
        return {
            "status": "success",
            "class_id": class_id,
            "result": sign_text
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}