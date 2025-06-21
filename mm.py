import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
import pandas as pd
from sklearn.metrics import classification_report

train_transform = transforms.Compose([
    transforms.Grayscale(1),              
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('C:/Users/gsr20/Desktop/GPU/data/train', transform=train_transform)
test_dataset = datasets.ImageFolder('C:/Users/gsr20/Desktop/GPU/data/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class BetterStarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 512),    
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)                
        )

    def forward(self, x):
        h = self.f_extractor(x)             
        f = h.view(h.size(0), -1)           
        return self.classifier(f)           

def train(device, epochs):
    model = BetterStarCNN().to(device)
    criterion = nn.CrossEntropyLoss()      
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)         
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)

    time_elapsed = time.time() - start_time
    return model, time_elapsed, 100 * total_correct / total_samples, total_loss / len(train_loader)

def evaluate(model, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            correct += (y_pred.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    return 100 * correct / total, time.time() - start_time

for E in [5, 10, 25, 50]:
    print(f"\n== EPOCHS: {E} ==")

    cpu = torch.device("cpu")
    cpu_model, cpu_train_time, cpu_acc, cpu_loss = train(cpu, E)
    cpu_test_acc, cpu_test_time = evaluate(cpu_model, cpu)

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        gpu_model, gpu_train_time, gpu_acc, gpu_loss = train(gpu, E)
        gpu_test_acc, gpu_test_time = evaluate(gpu_model, gpu)
    else:
        gpu_train_time = gpu_acc = gpu_loss = gpu_test_acc = gpu_test_time = "N/A"

    df = pd.DataFrame({
        "Device": ["CPU", "GPU"],
        "Epochs": [E, E],
        "Train Time (s)": [round(cpu_train_time, 2), round(gpu_train_time, 2) if gpu_train_time != "N/A" else "N/A"],
        "Test Time (s)": [round(cpu_test_time, 2), round(gpu_test_time, 2) if gpu_test_time != "N/A" else "N/A"],
        "Train Acc (%)": [round(cpu_acc, 2), round(gpu_acc, 2) if gpu_acc != "N/A" else "N/A"],
        "Test Acc (%)": [round(cpu_test_acc, 2), round(gpu_test_acc, 2) if gpu_test_acc != "N/A" else "N/A"],
        "Train Loss": [round(cpu_loss, 4), round(gpu_loss, 4) if gpu_loss != "N/A" else "N/A"]
    })
    print(df.to_string(index=False))
