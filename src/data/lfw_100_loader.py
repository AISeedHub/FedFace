"""LFW 100 FACE Dataset Loader"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class LFW100Dataset(Dataset):
    """LFW 100 FACE Dataset for Face Recognition
    - Load iamges from folder tree structure
    - Each subfolder corresponds to one identity
    - Use InsightFace to get face embeddings, so the returned is (embedding, label)
    """

    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or self._default_transform()

        # Read image and labels from data_dir
        self.data_dir = Path(data_dir)
        self.samples = []
        for folder in Path(data_dir).iterdir():
            if not folder.is_dir():
                continue
            for img_path in folder.iterdir():
                if not img_path.is_file():
                    continue
                label = folder.name
                # get last 2 iamges for val/test <--- simplified for testing
                self.samples.append((img_path, label))
                if split == "val":
                    if len(self.samples) > 2:
                        self.samples = self.samples[-2:]
                else:
                    if len(self.samples) > 8:
                        self.samples = self.samples[:-2]

        # Get embeddings using InsightFace
        for i in range(len(self.samples)):
            img_path, label = self.samples[i]
            # Dummy embedding for testing
            embedding = np.random.rand(1, 512).astype("float32")
            self.samples[i] = (embedding, label)

        # Load annotations (simplified)
        # self.samples = self._load_annotations()

    def _default_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _load_annotations(self):
        # annotations are folder names
        samples = []
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        embbeding, label = self.samples[idx]

        if self.transform:
            embbeding = self.transform(embbeding)

        return embbeding, label


def prepare_federated_data(num_clients: int, data_dir: str, batch_size: int = 8):
    """
    Partition WIDER FACE dataset for federated learning

    Returns:
        trainloaders: List of DataLoader for each client
        valloaders: List of validation DataLoader
        testloader: Global test DataLoader
    """

    # Load full dataset
    full_dataset = LFW100Dataset(data_dir, split="train")
    test_dataset = LFW100Dataset(data_dir, split="val")

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


if __name__ == "__main__":
    # Test the dataset loader
    data_dir = "E:\\dataset\\lfw\\lfw_100_aug"
    dataset = LFW100Dataset(data_dir, split="train")
    print(f"Dataset size: {len(dataset)}")
    for i in range(3):
        embedding, label = dataset[i]
        print(f"Sample {i}: Embedding shape: {embedding.shape}, Label: {label}")
