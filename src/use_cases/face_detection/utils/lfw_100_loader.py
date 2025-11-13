"""LFW 100 FACE Dataset Loader"""

import os

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_face_embedding(app, image: np.ndarray) -> np.ndarray:
    """
    Extracts a face embedding from an image using the InsightFace model.

    Args:
        app: An initialized InsightFace FaceAnalysis application.
        image (np.ndarray): The input image from which to extract the face embedding.

    Returns:
        np.ndarray: The extracted face embedding.
    """
    # For testing, return a random embedding
    embedding = app.get(image)[0]["embedding"]

    return embedding


def make_embedding_from_folder(data_dir: str, save_path: str = None):
    """Read images and labels from folder structure"""
    from insightface.app.face_analysis import FaceAnalysis

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    images = []
    labels = []
    for identity_folder in os.listdir(data_dir)[:10]:
        print(f"Processing identity: {identity_folder}")
        identity_path = os.path.join(data_dir, identity_folder)
        if not os.path.isdir(identity_path):
            continue
        samples = []
        for img_name in os.listdir(identity_path):
            print(f"Processing image: {img_name}", end="\r")
            img_path = os.path.join(identity_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            embedding = get_face_embedding(app, image)
            samples.append((embedding, identity_folder))
        images.extend([s[0] for s in samples])
        labels.extend([s[1] for s in samples])
    # save to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    if save_path:
        np.savez_compressed(
            save_path,
            images=images,
            labels=labels,
        )
    return images, labels


class LFW100EmbDataset(Dataset):
    """LFW 100 FACE Dataset for Face Recognition
    - Load embeddings from npz file
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None) -> None:
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or self._default_transform()

        # Read image and labels from npz
        data_path = os.path.join(data_dir, "data.npz")
        data = np.load(data_path)

        self.samples = list(zip(data["images"], data["labels"]))
        # encode labels to integers
        unique_labels = sorted(set(data["labels"]))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def _default_transform(self):
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        embedding, label = self.samples[idx]

        if self.transform:
            embedding = self.transform(embedding)

        return embedding, self.label_to_idx[label]


class LFW100Dataset(Dataset):
    """LFW 100 FACE Dataset for Face Recognition
    - Load iamges from folder tree structure
    - Each subfolder corresponds to one identity
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None) -> None:
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or self._default_transform()

        # Read image and labels from npz
        self.samples = []
        for identity_folder in os.listdir(data_dir):
            identity_path = os.path.join(data_dir, identity_folder)
            if not os.path.isdir(identity_path):
                continue
            for img_name in os.listdir(identity_path):
                img_path = os.path.join(identity_path, img_name)
                image = PIL.Image.open(img_path).convert("RGB")
                if image is None:
                    continue
                self.samples.append((image, identity_folder))

        unique_labels = sorted({s[1] for s in self.samples})
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def _default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image, label = self.samples[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.label_to_idx[label]


if __name__ == "__main__":
    # Test the dataset loader
    # data_dir = "E:/dataset/lfw/lfw_100_aug"
    # make_embedding_from_folder(data_dir, save_path="lfw_10.npz")
    dataset = LFW100Dataset(data_dir="E:/dataset/lfw/lfw_100_aug", split="train")
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        image, label = dataset[i]
        print(f"Sample {i}: Image shape: {image.shape}, Label: {label}")
