import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ── label map ────────────────────────────────────────────────────
label_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'}
class_names = [label_map[i] for i in sorted(label_map.keys())]

# ── load test data ────────────────────────────────────────────────
test_data = pd.read_csv("sign_mnist_test.csv")
X_test = torch.tensor(test_data.iloc[:,1:].values / 255, dtype=torch.float32)
y_test = torch.tensor(test_data.iloc[:,0].values, dtype=torch.long)
X_test = X_test.reshape(-1, 1, 28, 28)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# ── model definition (same as train.py) ──────────────────────────
class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(1600, 128)
        self.fc2   = nn.Linear(128, 25)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ── load saved weights ────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLModel().to(device)
model.load_state_dict(torch.load('asl_model.pth', weights_only=True))
model.eval()

# ── collect all predictions ───────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = torch.argmax(model(images), dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ── confusion matrix ──────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.4, linecolor='white')
plt.title("Confusion Matrix — ASL Hand Sign Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# ── classification report ─────────────────────────────────────────
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:\n")
print(report)

# save report as image too
fig, ax = plt.subplots(figsize=(8, 7))
ax.axis('off')
ax.text(0.01, 0.99, report, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace')
plt.title("Classification Report — ASL Hand Sign Recognition")
plt.tight_layout()
plt.savefig("classification_report.png", dpi=150)
plt.show()
print("Saved: classification_report.png")