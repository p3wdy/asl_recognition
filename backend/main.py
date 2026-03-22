from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import uvicorn
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600,128)
        self.fc2 = nn.Linear(128,25)
        self.relu = nn.ReLU()
    
    #process flow
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = ASLModel()
#gpu/cpu configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load('../model/asl_model.pth', weights_only=True))
model.eval()

label_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with torch.no_grad():
        image_pil = Image.open(file.file)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image_pil.convert('RGB')))
        result = detector.detect(mp_image)
        image_gray = image_pil.convert('L')
        if result.hand_landmarks:
            h, w = image_pil.height, image_pil.width
            landmarks = result.hand_landmarks[0]
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            x1 = int(max(0, min(x_coords) - 20))
            x2 = int(min(w, max(x_coords) + 20))
            y1 = int(max(0, min(y_coords) - 20))
            y2 = int(min(h, max(y_coords) + 20))
            image_gray = image_gray.crop((x1, y1, x2, y2))
        image = ImageOps.autocontrast(image_gray)
        image = image.resize((28, 28))
        image = np.array(image)
        image = image / 255
        image = torch.tensor(image, dtype=torch.float32)
        image = image.reshape(1, 1, 28, 28)
        image = image.to(device)
        predictions = model(image)
        predicted = torch.argmax(predictions)
        letter = label_map[predicted.item()]
        return {"letter": letter}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)