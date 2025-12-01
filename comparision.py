import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import time
import pandas as pd

class BetterStarCNN(nn.Module):
    def __init__(self):
        super(BetterStarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
0        return self.classifier(x)

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

train_dataset = ImageFolder('/data/train', transform=train_transform)
test_dataset = ImageFolder('/data/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_model(device, epochs):
    model = BetterStarCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    start_time = time.time()
    total_loss = 0.0

    correct = 0
    total = 0

    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return model, end_time - start_time, accuracy, avg_loss

def evaluate_model(model, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    report = classification_report(all_labels, all_preds, output_dict=True)
    return acc, report

for epoch_count in [5, 10, 25, 50]:
    print(f"\n===== Results for {epoch_count} Epochs =====")

    cpu_device = torch.device("cpu")
    cpu_model, cpu_time, cpu_train_acc, cpu_train_loss = train_model(cpu_device, epoch_count)
    cpu_test_acc, _ = evaluate_model(cpu_model, cpu_device)

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_model, gpu_time, gpu_train_acc, gpu_train_loss = train_model(gpu_device, epoch_count)
        gpu_test_acc, _ = evaluate_model(gpu_model, gpu_device)
    else:
        gpu_time = gpu_train_acc = gpu_train_loss = gpu_test_acc = "N/A"

    data = {
        "Device": ["CPU", "GPU"],
        "Training Time (s)": [round(cpu_time, 2), round(gpu_time, 2) if gpu_time != "N/A" else "N/A"],
        "Train Accuracy (%)": [round(cpu_train_acc, 2), round(gpu_train_acc, 2) if gpu_train_acc != "N/A" else "N/A"],
        "Test Accuracy (%)": [round(cpu_test_acc, 2), round(gpu_test_acc, 2) if gpu_test_acc != "N/A" else "N/A"],
        "Train Loss": [round(cpu_train_loss, 4), round(gpu_train_loss, 4) if gpu_train_loss != "N/A" else "N/A"]
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

