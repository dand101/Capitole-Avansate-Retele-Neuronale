from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = [(image, label) for image, label in dataset]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, label
