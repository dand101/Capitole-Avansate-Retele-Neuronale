import torch
from torch import nn, Tensor
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import optuna

device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True
scaler = GradScaler(device, enabled=enable_half)


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


augmentation_transforms = v2.Compose([
    v2.RandomRotation(degrees=5),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])

basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])

train_set = CIFAR100('/kaggle/input/fii-atnn-2024-assignment-2', download=True, train=True,
                     transform=augmentation_transforms)
test_set = CIFAR100('/kaggle/input/fii-atnn-2024-assignment-2', download=True, train=False,
                    transform=basic_transforms)
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


# Training loop
def train(model, optimizer):
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = functional.cross_entropy(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss


# Validation loop
@torch.inference_mode()
def val(model):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = functional.cross_entropy(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return accuracy, avg_loss


# Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    momentum = trial.suggest_uniform('momentum', 0.85, 0.99)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    # Define model, optimizer, and scheduler
    model = VGG16().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0

    for epoch in range(50):
        train_acc, train_loss = train(model, optimizer)
        val_acc, val_loss = val(model)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


# Run Optuna optimization
study = optuna.create_study(direction='maximize')

starting_point = {
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-2
}

study.enqueue_trial(starting_point)

study.optimize(objective, n_trials=50)

# Print the best hyperparameters and result
print('Best trial:')
trial = study.best_trial
print(f'  Best Validation Accuracy: {trial.value}')
print('  Best Hyperparameters: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
