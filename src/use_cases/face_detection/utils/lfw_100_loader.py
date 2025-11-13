"""LFW 100 FACE Dataset Loader"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_face_embedding(app, image: np.ndarray) -> np.ndarray:
    """
    Dummy function to get face embedding from image using InsightFace.
    In real implementation, this would use InsightFace model to extract embeddings.
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
    - Load iamges from folder tree structure
    - Each subfolder corresponds to one identity
    - Use InsightFace to get face embeddings, so the returned is (embedding, label)
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

        embbeding, label = self.samples[idx]

        if self.transform:
            embbeding = self.transform(embbeding)

        return embbeding, self.label_to_idx[label]


if __name__ == "__main__":
    # Test the dataset loader
    # data_dir = "E:/dataset/lfw/lfw_100_aug"
    # make_embedding_from_folder(data_dir, save_path="lfw_10.npz")
    dataset = LFW100EmbDataset(
        data_dir="use_cases/face_detection/distributed_data/client_0/", split="train"
    )
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        emb, label = dataset[i]
        print(f"Sample {i}: Embedding shape: {emb.shape}, Label: {label}")
