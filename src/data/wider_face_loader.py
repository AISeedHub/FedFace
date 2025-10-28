"""WIDER FACE Dataset Loader"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class WIDERFaceDataset(Dataset):
    """WIDER FACE Dataset for Face Detection"""

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or self._default_transform()

        # Load annotations (simplified)
        self.samples = self._load_annotations()

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_annotations(self):
        # Simplified: Load image paths and bounding boxes
        # In practice, parse WIDER FACE annotation format
        samples = []
        # TODO: Implement actual WIDER FACE parsing
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(bbox, dtype=torch.float32), label


def prepare_federated_data(num_clients: int, data_dir: str, batch_size: int = 8):
    """
    Partition WIDER FACE dataset for federated learning

    Returns:
        trainloaders: List of DataLoader for each client
        valloaders: List of validation DataLoader
        testloader: Global test DataLoader
    """

    # Load full dataset
    full_dataset = WIDERFaceDataset(data_dir, split='train')
    test_dataset = WIDERFaceDataset(data_dir, split='val')

    # Partition for clients
    partition_size = len(full_dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(full_dataset) - sum(lengths)

    datasets_split = random_split(
        full_dataset, lengths, torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    trainloaders = []
    valloaders = []

    for dataset in datasets_split:
        # Split train/val 80/20
        len_val = int(len(dataset) * 0.2)
        len_train = len(dataset) - len_val

        ds_train, ds_val = random_split(
            dataset, [len_train, len_val], torch.Generator().manual_seed(42)
        )

        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloaders, valloaders, testloader