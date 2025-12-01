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

# -------- Model Definition --------
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
        return self.classifier(x)

# -------- Dataset --------
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

# -------- Train Function --------
def train_model(device):
    model = BetterStarCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_losses = []
    train_accuracies = []

    start_time = time.time()

    for epoch in range(5):  # Only 5 epochs
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"[{device}] Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    end_time = time.time()

    return model, {
        "losses": train_losses,
        "accuracies": train_accuracies,
        "time": end_time - start_time
    }

# -------- Evaluate Function --------
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

# -------- Train on CPU --------
cpu_device = torch.device("cpu")
print("Training on CPU...")
cpu_model, cpu_metrics = train_model(cpu_device)
cpu_acc, cpu_report = evaluate_model(cpu_model, cpu_device)
print("CPU Done.\n")

# -------- Train on GPU (if available) --------
if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
    print("Training on GPU...")
    gpu_model, gpu_metrics = train_model(gpu_device)
    gpu_acc, gpu_report = evaluate_model(gpu_model, gpu_device)
    print("GPU Done.\n")
else:
    gpu_metrics = gpu_acc = gpu_report = None

# -------- Comparison Results --------
print("========= CPU vs GPU Comparison (5 Epochs) =========")
print(f"CPU Time: {cpu_metrics['time']:.2f}s | Accuracy: {cpu_acc:.2f}%")
if gpu_metrics:
    print(f"GPU Time: {gpu_metrics['time']:.2f}s | Accuracy: {gpu_acc:.2f}%")
else:
    print("GPU not available. Skipping GPU comparison.")

# -------- Plotting --------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cpu_metrics["losses"], label='CPU Loss')
if gpu_metrics:
    plt.plot(gpu_metrics["losses"], label='GPU Loss')
plt.title("Training Loss Comparision")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cpu_metrics["accuracies"], label='CPU Accuracy')
if gpu_metrics:
    plt.plot(gpu_metrics["accuracies"], label='GPU Accuracy')
plt.title("Training Accuracy Comparision")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

