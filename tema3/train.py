# train.py
from datasets import get_data_loaders
from config import load_config
import torch


def main():
    config = load_config()
    train_loader, test_loader = get_data_loaders(config)

    print("Loaded Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    for images, labels in train_loader:
        print(f"Batch of images has shape {images.shape}")
        print(f"Batch of labels has shape {labels.shape}")
        break


if __name__ == '__main__':
    main()
