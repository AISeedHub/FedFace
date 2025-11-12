"""Prepare real face detection dataset using publicly available data"""

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CelebA
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def download_celeba(data_root="use_cases/face_detection/raw_data"):
    """
    Download CelebA dataset (a subset for face detection).
    CelebA contains 200k+ celebrity face images with annotations.

    Note: First download requires ~1.4GB
    """
    print("=" * 80)
    print("📥 Downloading CelebA Dataset for Face Detection")
    print("=" * 80)
    print("\n⚠️  This may take a while on first run (~1.4GB download)")
    print("Dataset will be cached for future use.\n")

    os.makedirs(data_root, exist_ok=True)

    # Define transforms for face images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64 for faster training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Download CelebA dataset
        # target_type='attr' gives us 40 binary attributes including face detection-relevant features
        print("📥 Downloading CelebA dataset...")
        dataset = CelebA(
            root=data_root,
            split='train',
            target_type='attr',
            transform=transform,
            download=True
        )
        print(f"\n✅ Dataset downloaded successfully!")
        print(f"   - Total images: {len(dataset)}")
        print(f"   - Image size: 64x64x3")
        print(f"   - Attributes: {len(dataset.attr_names)}")

        return dataset

    except Exception as e:
        print(f"\n❌ Error downloading CelebA: {e}")
        print("\n💡 Alternative: Using synthetic data for demonstration")
        return None


def create_synthetic_face_data(num_images=1000, image_size=64):
    """
    Create synthetic face-like data for quick testing.
    This is a fallback if real dataset download fails.
    """
    print("\n🎨 Creating synthetic face detection data...")
    print(f"   - Images: {num_images}")
    print(f"   - Size: {image_size}x{image_size}x3\n")

    # Create synthetic images (random noise with face-like structure)
    images = torch.randn(num_images, 3, image_size, image_size) * 0.5 + 0.5

    # Create binary labels (face/no-face) for demonstration
    labels = torch.randint(0, 2, (num_images,))

    return images, labels


def prepare_face_detection_data(
        use_real_data=True,
        num_synthetic_images=1000,
        data_root="use_cases/face_detection/raw_data"
):
    """
    Prepare face detection dataset.

    Args:
        use_real_data: If True, try to download CelebA. If False, use synthetic data.
        num_synthetic_images: Number of synthetic images if real data not available
        data_root: Root directory for data

    Returns:
        images, labels tensors
    """

    if use_real_data:
        dataset = download_celeba(data_root)

        if dataset is not None:
            print("\n🔄 Converting dataset to tensors...")

            # For face detection, we'll use a subset and create binary labels
            # based on certain attributes (e.g., presence of certain facial features)
            num_samples = min(1000, len(dataset))  # Use 1000 images for demo

            images_list = []
            labels_list = []

            for i in tqdm(range(num_samples), desc="Processing images"):
                img, attrs = dataset[i]
                images_list.append(img)

                # Create binary label: face present (always 1 in CelebA) or use attribute
                # For demo, we'll use "Smiling" attribute as binary classification
                label = attrs[31].item()  # Index 31 is "Smiling" attribute
                labels_list.append(label)

            images = torch.stack(images_list)
            labels = torch.tensor(labels_list)

            print(f"\n✅ Processed {len(images)} real face images")
            print(f"   - Positive samples (smiling): {labels.sum().item()}")
            print(f"   - Negative samples: {(1 - labels).sum().item()}")

            return images, labels

    # Fallback to synthetic data
    print("\n⚠️  Using synthetic data (set use_real_data=True for real faces)")
    return create_synthetic_face_data(num_synthetic_images)


if __name__ == "__main__":
    # Test the data preparation
    images, labels = prepare_face_detection_data(use_real_data=True)
    print(f"\n📊 Data shape: {images.shape}")
    print(f"📊 Labels shape: {labels.shape}")