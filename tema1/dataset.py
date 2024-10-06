from torchvision.datasets import MNIST
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_images(images):
    img = []
    labels = []
    for i in range(len(images)):
        image, label = images[i]
        img_tensor = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).view(-1)
        img.append(img_tensor)
        label_tensor = torch.tensor(label, dtype=torch.int64)
        labels.append(label_tensor)
    return torch.stack(img), torch.tensor(labels)


def load_data():
    train_dataset = MNIST(root='./data', train=True, download=True)
    test_dataset = MNIST(root='./data', train=False, download=True)

    # print(train_dataset[130][1])
    train_x, train_y = process_images(train_dataset)
    test_x, test_y = process_images(test_dataset)
    train_x, val_x = train_x[:55000], train_x[55000:]
    train_y, val_y = train_y[:55000], train_y[55000:]

    return (train_x.to(device), train_y.to(device)), (val_x.to(device), val_y.to(device)), (
        test_x.to(device), test_y.to(device))
