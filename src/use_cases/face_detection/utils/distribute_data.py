"""Distribute face detection data to federated clients"""

import argparse
import os
import shutil

import numpy as np


def distribute_data(
    images, labels, num_clients=2, output_path="./", non_iid=True, alpha=0.5
):
    """
    Distribute data to clients in IID or Non-IID fashion.

    Args:
        images: Tensor of images
        labels: Tensor of labels
        num_clients: Number of clients
        output_path: Output directory
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

        # Save as npz files
        np.savez_compressed(
            os.path.join(client_dir, "data.npz"),
            images=client_images,
            labels=client_labels,
        )

        # Calculate label distribution
        unique, counts = np.unique(client_labels, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))

        print(f"✓ Client {client_id}:")
        print(f"   - Images: {len(idx):4d} ({len(idx)/num_images*100:.1f}%)")
        print(f"   - Label distribution: {label_dist}")
        print(f"   - Saved to: {client_dir}")

    print("-" * 80)
    print("\n✅ Data distribution complete!")
    print(f"📁 Data saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and distribute face detection data"
    )
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients")
    # parser.add_argument(
    #     "--num-images",
    #     type=int,
    #     default=1000,
    #     help="Number of images (for synthetic data)",
    # )
    # parser.add_argument(
    #     "--use-real-data", action="store_true", help="Use real CelebA data"
    # )
    parser.add_argument(
        "--dir", type=str, default="./lfw_100.npz", help="Path to data npz file"
    )
    parser.add_argument(
        "--non-iid", action="store_true", default=True, help="Use non-IID distribution"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="use_cases/face_detection/distributed_data",
        help="Output directory",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("🌸 FedFlower - Face Detection Data Preparation")
    print("=" * 80)

    # Step 1: Prepare dataset
    print("\n📥 Step 1: Preparing dataset...")
    # images, labels = prepare_face_detection_data(
    #     use_real_data=args.use_real_data, num_synthetic_images=args.num_images
    # )
    data = np.load(args.dir)
    images, labels = data["images"], data["labels"]

    # Step 2: Distribute to clients
    print(f"\n📤 Step 2: Distributing to {args.num_clients} clients...")
    distribute_data(
        images=images,
        labels=labels,
        num_clients=args.num_clients,
        output_path=args.output,
        non_iid=args.non_iid,
    )

    print("\n" + "=" * 80)
    print("✅ All done! You can now start the federated learning process.")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Start server: python use_cases/face_detection/main_server.py")
    print(
        "  2. Start clients: python use_cases/face_detection/main_client.py --client-id 0"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
