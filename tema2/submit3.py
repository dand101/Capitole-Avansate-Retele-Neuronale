import torch
from torch import nn, Tensor
import torch.nn.functional as functional
from torchvision.datasets import CIFAR100
import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2, AutoAugment, AutoAugmentPolicy
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True
scaler = GradScaler(device, enabled=enable_half)

print(device)
EPOC = 50


def create_plots(f, train_accuracies, val_accuracies, train_losses, val_losses, epochs):
    if f is True:
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


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def apply_transformations(image, label, transform_type):
    if transform_type == 'flip':
        transform = v2.RandomHorizontalFlip(p=1.0)
        return transform(image), label
    elif transform_type == 'jitter':
        transform = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        return transform(image), label
    elif transform_type == 'crop':
        transform = v2.RandomResizedCrop(size=(32, 32), scale=(0.6, 1.0), ratio=(0.75, 1.33))
        return transform(image), label
    elif transform_type == 'rotation':
        transform = v2.RandomRotation(degrees=15)
        return transform(image), label
    elif transform_type == 'perspective':
        transform = v2.RandomPerspective(distortion_scale=0.5, p=1.0)
        return transform(image), label
    elif transform_type == 'affine':
        transform = v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
        return transform(image), label
    elif transform_type == 'grayscale':
        transform = v2.RandomGrayscale(p=1.0)
        return transform(image), label
    elif transform_type == 'solarize':
        transform = v2.RandomSolarize(threshold=128)
        return transform(image), label
    else:
        return image, label


def create_extra(base_train_set):
    def normalize(image):
        return v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)(image)

    def to_tensor(image):
        return v2.ToTensor()(image)

    images = base_train_set.data
    labels = base_train_set.targets

    train_set = []
    for img, label in zip(images, labels):
        img = Image.fromarray(img)

        tensor_image = to_tensor(img)
        normalized_image = normalize(tensor_image)
        train_set.append((normalized_image, label))

        # flipped_image, label = apply_transformations(img, label, 'flip')
        # train_set.append((normalize(to_tensor(flipped_image)), label))
        #
        # cropped_image, label = apply_transformations(img, label, 'crop')
        # train_set.append((normalize(to_tensor(cropped_image)), label))
        #
        # affine_image, label = apply_transformations(img, label, 'affine')
        # train_set.append((normalize(to_tensor(affine_image)), label))
        #
        # rotated_image, label = apply_transformations(img, label, 'rotation')
        # train_set.append((normalize(to_tensor(rotated_image)), label))
        #
        # perspective_image, label = apply_transformations(img, label, 'perspective')
        # train_set.append((normalize(to_tensor(perspective_image)), label))
        #
        # jittered_image, label = apply_transformations(img, label, 'jitter')
        # train_set.append((normalize(to_tensor(jittered_image)), label))
        #
        # grayscale_image, label = apply_transformations(img, label, 'grayscale')
        # train_set.append((normalize(to_tensor(grayscale_image)), label))
        #
        # solarized_image, label = apply_transformations(img, label, 'solarize')
        # train_set.append((normalize(to_tensor(solarized_image)), label))

    final_train_set = SimpleCachedDataset(train_set)

    return final_train_set


basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])

train_set = CIFAR100('/kaggle/input/fii-atnn-2024-assignment-2', download=True, train=True, transform=None)
test_set = CIFAR100('/kaggle/input/fii-atnn-2024-assignment-2', download=True, train=False, transform=basic_transforms)

train_set = create_extra(train_set)

test_set = SimpleCachedDataset(test_set)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=pin_memory)
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

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2, nesterov=True, fused=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

cutMix = v2.CutMix(num_classes=100, alpha=1.0)
mixUp = v2.MixUp(num_classes=100, alpha=1.0)

rand_choice = v2.RandomChoice([cutMix, mixUp])


def train():
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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


best_model_state = model.state_dict()
best = 0.0
epochs = list(range(EPOC))
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc, train_loss = train()

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        val_acc, val_loss = val()
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        scheduler.step()

        if val_acc > best:
            best = val_acc
            torch.save(best_model_state, 'best_model.pth')
        tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.6f}, Best: {best:.6f}")

create_plots(True, train_accuracies, val_accuracies, train_losses, val_losses, epochs)
model.load_state_dict(torch.load('best_model.pth'))

data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv(f"submission_new_{time.time()}.csv", index=False)
