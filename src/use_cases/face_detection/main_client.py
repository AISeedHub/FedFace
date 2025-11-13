"""Main client for face classification federated learning"""

import argparse
import os
import sys

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.fed_core.fed_client import FedFlowerClient
from src.use_cases.face_detection.models.cnn import SimpleCNN
from src.use_cases.face_detection.models.mlp import MLP
from src.use_cases.face_detection.models.resnet import PretrainedResNet
from src.use_cases.face_detection.utils.lfw_100_loader import (
    LFW100Dataset,
    LFW100EmbDataset,
)

MODELS = {
    "mlp": MLP,
    "resnet": PretrainedResNet,
    "cnn": SimpleCNN,
}

LOADER = {
    "folder": LFW100Dataset,
    "npz": LFW100EmbDataset,
}


class FaceClassificationClient(FedFlowerClient):
    """Face Classification Client for Federated Learning"""

    def __init__(self, client_id: int, config: dict):
        super().__init__(client_id, config)

        # Initialize model
        self.model = MODELS[config["model"]["name"]](
            num_classes=config["model"]["num_classes"]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["learning_rate"], momentum=0.9
        )

        # Load client data
        self.train_loader, self.test_loader = self._load_data()

    def _load_data(self):
        """Load client-specific data"""
        data_path = os.path.join(
            self.config["distributed_data_path"], f"client_{self.client_id}"
        )

        dataset = LOADER[self.config["data_type"]](data_dir=data_path, split="train")
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["batch_size"], shuffle=False
        )

        return train_loader, test_loader

    def train_model(self, epochs: int) -> dict[str, float]:
        """Train model locally"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        print(
            f"[Client {self.client_id}] Initialized with {len(self.train_loader.dataset)} training samples"
        )
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            total_loss += epoch_loss
            print(
                f"[Client {self.client_id}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(self.train_loader):.4f}"
            )

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / (epochs * len(self.train_loader))

        print(
            f"[Client {self.client_id}] Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    def evaluate_model(self) -> tuple[float, float, dict]:
        """Evaluate model"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        print(
            f"[Client {self.client_id}] Initialized with {len(self.test_loader.dataset)} testing samples"
        )
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = test_loss / len(self.test_loader)

        print(
            f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        return avg_loss, accuracy, {"test_accuracy": accuracy}

    def get_model_parameters(self) -> list[np.ndarray]:
        """Get model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def _get_dataset_size(self) -> int:
        """Return size of local training dataset"""
        return len(self.train_loader.dataset)


def load_config(config_path="src/use_cases/face_detection/configs/base.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def main():
    """Start federated learning client"""
    parser = argparse.ArgumentParser(description="Face Classification Federated Client")
    parser.add_argument(
        "--client-id", type=int, required=True, help="Client ID (0, 1, ...)"
    )
    parser.add_argument(
        "--server-address", type=str, default="127.0.0.1:9000", help="Server address"
    )
    args = parser.parse_args()

    print(f"🌸 FedFlower - Face Classification Client {args.client_id}")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Create client
    client = FaceClassificationClient(args.client_id, config)

    print(f"🚀 Connecting to server at {args.server_address}")
    print("=" * 50)

    # Start client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
