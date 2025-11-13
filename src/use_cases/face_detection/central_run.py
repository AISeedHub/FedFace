import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.use_cases.face_detection.models.mlp import MLP
from src.use_cases.face_detection.utils.lfw_100_loader import LFW100EmbDataset


class FaceClassification:
    """Face Classification"""

    def __init__(self, config: dict):

        # Initialize model
        self.model = MLP(num_classes=config["model"]["num_classes"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.config = config

        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["learning_rate"], momentum=0.9
        )

        # Load client data
        self.train_loader, self.test_loader = self._load_data()

        print(f"Initialized with {len(self.train_loader.dataset)} training samples")

    def _load_data(self):
        """Load client-specific data"""
        data_path = os.path.join(self.config["data_path"])

        dataset = LFW100EmbDataset(data_dir=data_path, split="train")
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
                f" Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(self.train_loader):.4f}"
            )

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / (epochs * len(self.train_loader))

        print(f" Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    def evaluate_model(self) -> tuple[float, float, dict]:
        """Evaluate model"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

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

        print(f" Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy, {"test_accuracy": accuracy}


def load_config(config_path="src/use_cases/face_detection/configs/base.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def main():
    """Start federated learning client"""

    print("🌸 FedFlower - Face Classification Central")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Create client
    central_run = FaceClassification(config)
    central_run.train_model(epochs=config["local_epochs"])
    central_run.evaluate_model()

    print("=" * 50)


if __name__ == "__main__":
    main()
