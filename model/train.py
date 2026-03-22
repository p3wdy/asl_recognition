import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#training and testing data
train_data = pd.read_csv("sign_mnist_train.csv")
test_data = pd.read_csv("sign_mnist_test.csv")

'''print("The train data size is \n", train_data.shape)
print("The test data size is \n", test_data.shape)
print("train head \n", train_data.head())
print("test head \n", test_data.head())'''

#dictionary mapping
label_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'}
X_train = train_data.iloc[:,1:]
y_train = train_data.iloc[:,0]
X_test = test_data.iloc[:,1:]
y_test = test_data.iloc[:,0]

X_train = X_train/255
X_test = X_test/255

X_train = torch.tensor(X_train.values, dtype= torch.float32)
y_train = torch.tensor(y_train.values, dtype= torch.long)
X_test = torch.tensor(X_test.values, dtype= torch.float32)
y_test = torch.tensor(y_test.values, dtype= torch.long)

# print(y_train.min(), y_train.max())
# print(y_test.min(), y_test.max())

#reshaping to 2d from 1d //-1 does auto assign
X_train = X_train.reshape(-1, 1, 28, 28) #27455 images
X_test = X_test.reshape(-1, 1, 28, 28) #7172 images

#tensordatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

#dataloaders
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

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
#print(model)

#gpu/cpu configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
#print("Using device:", device)

#calculating loss values
criterion = nn.CrossEntropyLoss()
#updating weights
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# +++ NEW: lists to track loss and accuracy per epoch
train_losses = []
val_accuracies = []

#training model
for epoch in range(10):
    epoch_loss = 0                                          # +++ NEW
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()                           # +++ NEW
    
    train_losses.append(epoch_loss / len(train_loader))     # +++ NEW: save average loss

    # +++ NEW: evaluate accuracy at end of each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            predicted = torch.argmax(predictions, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracies.append((correct / total) * 100)          # +++ NEW: save accuracy
    model.train()                                           # +++ NEW: back to train mode

    print(f"Epoch {epoch+1}, Loss: {train_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")

#evaluating model (final evaluation, same as before)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        predicted = torch.argmax(predictions, dim = 1)
        total+= labels.size(0)
        correct+= (predicted == labels).sum().item()
    print(f"Test Accuracy: {(correct/total)*100:.2f}%")
model.train()

torch.save(model.state_dict(), 'asl_model.pth')
print("Model Saved Successfully")

# +++ NEW: plot and save training loss curve
epochs = range(1, 11)
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_losses, marker='o', color='steelblue')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()

# +++ NEW: plot and save validation accuracy curve
plt.figure(figsize=(6, 4))
plt.plot(epochs, val_accuracies, marker='o', color='seagreen')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("validation_accuracy.png", dpi=150)
plt.show()

'''
for i in range (10):
    tr_data = train_data.iloc[i,1:].values.reshape(28, 28) #selecting location and iterating and reshaping the size to the number of pixels
    plt.subplot(2,5,i+1)#row,column,iteration
    plt.imshow(tr_data,  cmap='gray')
    plt.title(label_map[train_data.iloc[i,0]])
    plt.axis(False)
plt.show()
plt.figure(figsize=(20,8))#inches or size
'''

'''
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)
'''