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
model.load_state_dict(torch.load("traffic_sign_cnn_2.pth", map_location=device))
model.to(device)
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
    4: "Ограничение скорости 70 км/ч",
    5: "Ограничение скорости 80 км/ч",
    6: "Конец зоны ограничения скорости 80 км/ч",
    7: "Ограничение скорости 100 км/ч",
    8: "Ограничение скорости 120 км/ч",
    9: "Обгон запрещен",
    10: "Обгон грузовым автомобилям запрещен",
    11: "Перекресток со второстепенной дорогой",
    12: "Главная дорога",
    13: "Уступите дорогу",
    14: "Движение без остановки запрещено (STOP)",
    15: "Движение запрещено",
    16: "Движение грузовых автомобилей запрещено",
    17: "Въезд запрещен (Кирпич)",
    18: "Прочие опасности",
    19: "Опасный поворот налево",
    20: "Опасный поворот направо",
    21: "Опасные повороты",
    22: "Неровная дорога",
    23: "Скользкая дорога",
    24: "Сужение дороги справа",
    25: "Дорожные работы",
    26: "Светофорное регулирование",
    27: "Пешеходный переход",
    28: "Дети",
    29: "Велосипедная дорожка",
    30: "Гололед/Снег",
    31: "Дикие животные",
    32: "Конец всех ограничений",
    33: "Движение направо",
    34: "Движение налево",
    35: "Движение прямо",
    36: "Движение прямо или направо",
    37: "Движение прямо или налево",
    38: "Объезд препятствия справа",
    39: "Объезд препятствия слева",
    40: "Круговое движение",
    41: "Конец зоны запрещения обгона",
    42: "Конец зоны запрещения обгона грузовым автомобилям"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
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