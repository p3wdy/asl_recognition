from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import uvicorn
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#model creation
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

label_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with torch.no_grad():
        image = Image.open(file.file).convert('L')
        # Improve contrast to match training data style
        image = ImageOps.autocontrast(image)
        image = image.resize((28, 28))
        image = np.array(image)
        image = image/255
        image = torch.tensor(image, dtype=torch.float32)
        image = image.reshape(1, 1, 28, 28)
        image = image.to(device)
        predictions = model(image)
        predicted = torch.argmax(predictions)
        letter = label_map[predicted.item()]
        return {"letter": letter}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)