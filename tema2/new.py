import torch
from torch import nn, Tensor
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True
scaler = GradScaler(device, enabled=enable_half)


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])

augmentation_transforms = v2.Compose([

    v2.RandomRotation(degrees=5),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    v2.ToImage(),

    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])

train_set = CIFAR100('/kaggle/input/fii-atnn-2024-assignment-2', download=True, train=True,
                     transform=basic_transforms)

test_set = CIFAR100('/kaggle/input/fii-atnn-2024-assignment-2', download=True, train=False, transform=basic_transforms)
train_set = SimpleCachedDataset(train_set)
test_set = SimpleCachedDataset(test_set)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


model = VGG16().to(device)
model = torch.jit.script(model)

criterion = nn.CrossEntropyLoss()

EPOC = 100

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2, nesterov=True, fused=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

cutMix = v2.CutMix(num_classes=100)
mixUp = v2.MixUp(num_classes=100)

rand_choice = v2.RandomChoice([cutMix, mixUp])


def train():
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # inputs, targets = cutMix(inputs, targets)
        inputs, targets = rand_choice(inputs, targets)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets.argmax(1)).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss


@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return accuracy, avg_loss


@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


best = 0.0
epochs = list(range(EPOC))
one_cycle_epochs = int(EPOC * 0.2)

with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc, train_loss = train()
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        scheduler.step()

        val_acc, val_loss = val()
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        if val_acc > best:
            best = val_acc
        tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o')
plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker='x')
plt.title("Training and Validation Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Validation Loss", marker='x')
plt.title("Training and Validation Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.grid(True)

plt.tight_layout()
plt.show()

data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)

df.to_csv(f"submission.csv_{time.time()}", index=False)
