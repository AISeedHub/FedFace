"""Distribute face detection data to federated clients"""

import os
import shutil
import yaml
import wget

import numpy as np


def load_npz_data(npz_path):
    """Load images and labels from npz file"""
    data = np.load(npz_path)
    images = data["images"]
    labels = data["labels"]
    return images, labels


def save_npz_data(images, labels, save_path):
    """Save images and labels to npz file"""
    np.savez_compressed(
        os.path.join(save_path, "data.npz"),
        images=images,
        labels=labels,
    )


def load_folder_data(folder_path):
    images = []
    labels = []
    for identity_folder in os.listdir(folder_path):
        identity_path = os.path.join(folder_path, identity_folder)
        if not os.path.isdir(identity_path):
            continue
        for img_name in os.listdir(identity_path):
            img_path = os.path.join(identity_path, img_name)
            images.append(img_path)
            labels.append(identity_folder)
    return np.array(images), np.array(labels)


def save_folder_data(images, labels, save_path):
    """Save images to folder structure"""
    for img, label in zip(images, labels):
        label_folder = os.path.join(save_path, label)
        os.makedirs(label_folder, exist_ok=True)
        img_name = os.path.basename(img)
        shutil.copy(img, os.path.join(label_folder, img_name))


def distribute_data(
        images,
        labels,
        num_clients=2,
        output_path="./",
        data_type="npz",
        non_iid=True,
        alpha=0.5,
):
    """
    Distribute data to clients in IID or Non-IID fashion.

    Args:
        images: list of images
        labels: list of labels
        num_clients: Number of clients
        output_path: Output directory
        data_type: Type of data ("npz" or "folder")
        non_iid: If True, create non-IID distribution
        alpha: Dirichlet distribution parameter (smaller = more non-IID)
    """

    num_images = len(images)
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    print("\n" + "=" * 80)
    print(f"📊 Distributing {num_images} images to {num_clients} clients")
    print(f"   Distribution: {'Non-IID' if non_iid else 'IID'}")
    print("=" * 80)

    # Remove old data
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Distribute data
    if non_iid:
        # Non-IID: Use Dirichlet distribution or simple imbalance
        if num_clients == 2:
            # Simple 80-20 split for 2 clients
            split_point = int(0.8 * num_images)
            client_indices = {0: indices[:split_point], 1: indices[split_point:]}
            print("\n   Split ratio: 80% - 20%")
        else:
            # Use Dirichlet distribution for more clients
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * num_images).astype(int)[:-1]
            client_indices = {
                i: indices[start:end]
                for i, (start, end) in enumerate(
                    zip([0] + list(proportions), list(proportions) + [num_images])
                )
            }
    else:
        # IID: Equal distribution
        split_indices = np.array_split(indices, num_clients)
        client_indices = {i: split for i, split in enumerate(split_indices)}

    # Save data for each client
    print("\n" + "-" * 80)
    for client_id, idx in client_indices.items():
        client_dir = os.path.join(output_path, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        client_images = images[idx]
        client_labels = labels[idx]

        if data_type == "npz":
            save_npz_data(client_images, client_labels, client_dir)
        elif data_type == "folder":
            save_folder_data(client_images, client_labels, client_dir)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Calculate label distribution
        unique, counts = np.unique(client_labels, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))

        print(f"✓ Client {client_id}:")
        print(f"   - Images: {len(idx):4d} ({len(idx) / num_images * 100:.1f}%)")
        print(f"   - Label distribution: {label_dist}")
        print(f"   - Saved to: {client_dir}")

    print("-" * 80)
    print("\n✅ Data distribution complete!")
    print(f"📁 Data saved to: {output_path}\n")


def main():
    # load configurations in yaml
    with open("src/use_cases/face_detection/configs/base.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("🌸 FedFlower - Face Detection Data Preparation")
    print("=" * 80)

    # Step 1: Prepare dataset
    print("\n📥 Step 1: Preparing dataset...")
    if not os.path.exists(config["full_data_path"]):
        print(f"❌ Dataset not found at {config['full_data_path']}. Starting to download dataset ...")
        try:
            if config["data_type"] == "folder":
                folder_path = config["full_data_path"]
                os.makedirs(folder_path, exist_ok=True)
                # use wget to download zip file and unzip it
                zip_url = config["dataset_url"]
                zip_path = os.path.join(folder_path, "dataset.zip")
                wget.download(zip_url, zip_path)
                shutil.unpack_archive(zip_path, folder_path)
                os.remove(zip_path)
                print(f"✅ Dataset downloaded and extracted to {folder_path}")
            else:
                raise ValueError(f"Unsupported data type: {config['data_type']}")
        except Exception as e:
            print(f"\n❌ Failed to download dataset: {e}")
            return

    if config["data_type"] == "npz":
        images, labels = load_npz_data(config["full_data_path"])
    elif config["data_type"] == "folder":
        # Add folder data loading method if needed
        images, labels = load_folder_data(config["train_data_path"])
    else:
        # Add other data loading methods if needed
        raise ValueError(f"Unsupported data type: {config["data_type"]}")

    # Step 2: Distribute to clients
    print(f"\n📤 Step 2: Distributing to {config["num_clients"]} clients...")
    distribute_data(
        images=images,
        labels=labels,
        num_clients=config["num_clients"],
        output_path=config["distributed_data_path"],
        non_iid=config["non_iid"],
        alpha=config["alpha"],
        data_type=config["data_type"],
    )

    print("\n" + "=" * 80)
    print("✅ All done! You can now start the federated learning process.")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Start server: python src/use_cases/face_detection/main_server.py")
    print(
        "  2. Start clients: python src/use_cases/face_detection/main_client.py --client-id 0"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
