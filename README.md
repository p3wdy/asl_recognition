# ASL Hand Sign Recognition System

A real-time American Sign Language (ASL) hand sign recognition system built with PyTorch, MediaPipe, and FastAPI. The system identifies static ASL alphabet signs (A–Y, excluding J and Z) from a live webcam feed and displays the predicted letter in a browser-based frontend.

---

## Demo

The system captures webcam frames in the browser, sends them to a FastAPI backend, detects the hand region using MediaPipe HandLandmarker, and classifies the gesture using a custom-trained CNN.

---

## Project Structure

```
asl_recognition/
├── venv/                               # Virtual environment (not tracked)
├── model/
│   ├── train.py                        # CNN training script
│   ├── evaluate.py                     # Confusion matrix and classification report
│   └── asl_model.pth                   # Saved model weights (not tracked)
├── backend/
│   ├── main.py                         # FastAPI backend with /predict endpoint
│   ├── main_backup.py                  # Backup version without MediaPipe
│   └── hand_landmarker.task            # MediaPipe model asset
├── frontend/
│   └── index.html                      # Webcam capture and prediction display
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Sign Language MNIST** — published by TECHPERSON on Kaggle.
A drop-in replacement for the classic MNIST dataset adapted for ASL hand gesture recognition.

- Training samples: 27,455
- Test samples: 7,172
- Classes: 24 (A–Y, J and Z excluded as they require motion)
- Image size: 28×28 grayscale
- Format: CSV (pixel values as columns)

Download from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Place both CSV files inside `model/` before training.

---

## Label Map

```python
label_map = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H',
    8:'I', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P',
    16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W',
    23:'X', 24:'Y'
}
```

> Note: Label 9 is absent from the dataset (corresponds to J which requires motion). Label 25 (Z) is also excluded. The label map skips these indices accordingly.

---

## Model Architecture

A custom CNN implemented in PyTorch:

| Layer | Details |
|---|---|
| Input | 1 × 28 × 28 grayscale image |
| Conv1 | 32 filters, 3×3 kernel → ReLU → MaxPool (2×2) |
| Conv2 | 64 filters, 3×3 kernel → ReLU → MaxPool (2×2) |
| Flatten | 64 × 5 × 5 = 1600 |
| FC1 | 1600 → 128, ReLU |
| FC2 | 128 → 25 output classes |

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 10
- **Batch size:** 64
- **Test accuracy:** 91.27%

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/p3wdy/asl_recognition.git
cd asl_recognition
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Sign Language MNIST dataset from Kaggle and place both CSV files inside `model/`:

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

### 5. Download MediaPipe hand landmark model

Download `hand_landmarker.task` from the MediaPipe documentation and place it inside `backend/`.

https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

---

## Training

```bash
cd model
python train.py
```

This will:
- Train the CNN on Sign Language MNIST for 10 epochs
- Print loss and validation accuracy per epoch
- Save model weights to `model/asl_model.pth`

---

## Evaluation

```bash
cd model
python evaluate.py
```

Generates:
- `confusion_matrix.png` — per-class prediction heatmap
- `classification_report.png` — precision, recall, F1-score per class

---

## Running the Application

### 1. Start the backend

```bash
cd backend
python main.py
```

The FastAPI server will start at `http://localhost:8000`.

### 2. Open the frontend

Open `frontend/index.html` directly in your browser.

### 3. Use the app

- Allow webcam access when prompted
- Click **Start Live** to begin real-time predictions
- Hold up an ASL hand sign in front of your webcam
- The predicted letter will appear on screen
- Click **Stop Live** to pause

---

## API

### POST `/predict`

Accepts a webcam frame and returns the predicted ASL letter.

**Request:** `multipart/form-data` with a JPEG image file

**Response:**
```json
{
  "letter": "A"
}
```

---

## Known Limitations

- **Static signs only** — J and Z require motion and are not supported
- **Single hand** — only one hand is detected at a time
- **Train-test distribution mismatch** — the Sign Language MNIST dataset uses heavily processed images that differ from real webcam input, which reduces live prediction accuracy compared to the 91.27% test accuracy
- **No word/sentence construction** — the system predicts individual letters only

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.5.1 + CUDA |
| Hand Detection | MediaPipe 0.10.x (Tasks API) |
| Backend | FastAPI + Uvicorn |
| Image Processing | Pillow |
| Frontend | HTML5 / JavaScript |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Hardware Used

- CPU: Intel Core i7-11800H
- GPU: NVIDIA RTX 3050Ti (CUDA 12.1)
- RAM: 16GB
- OS: Windows 11

---

## References

1. Fels, S. S. and Hinton, G. E. (1993). Glove-Talk: A Neural Network Interface between a Data-Glove and a Speech Synthesizer. IEEE Transactions on Neural Networks.
2. Pavlovic, V. I., Sharma, R. and Huang, T. S. (1997). Visual Interpretation of Hand Gestures for Human-Computer Interaction: A Review. IEEE TPAMI.
3. Krizhevsky, A., Sutskever, I. and Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.
4. TECHPERSON (2017). Sign Language MNIST. Kaggle. https://www.kaggle.com/datasets/datamunge/sign-language-mnist
5. Lugaresi, C. et al. (2019). MediaPipe: A Framework for Perceiving and Processing Reality. IEEE CVPR Workshop.
6. Zhang, F. et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. ECCV Workshop.
7. Ramírez, S. (2019). FastAPI. https://fastapi.tiangolo.com

---

## Author

**Rithul Rajith Kumar**  
