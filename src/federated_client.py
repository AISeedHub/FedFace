"""Face Detection Federated Client"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

# Import from fedflower-core
from fedflower.client import FedFlowerClient

from models.mobilenet_ssd import MobileNetV3_SSD


class FaceDetectionClient(FedFlowerClient):
    """
    Face Detection Client for Federated Learning
    Inherits from FedFlowerClient (fedflower-core)
    """

    def __init__(
            self,
            client_id: int,
            trainloader,
            valloader,
            config: Dict,
            device: str = "cpu"
    ):
        super().__init__(client_id, config)

        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device(device)

        # Initialize model
        self.model = MobileNetV3_SSD(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum']
        )

        print(f"📱 Client {client_id} initialized")
        print(f"   Model size: {self.model.get_model_size_mb():.2f} MB")

    def train_model(self, epochs: int) -> Dict[str, float]:
        """Train face detection model locally."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for images, bboxes, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                bbox_pred, class_pred = self.model(images)

                # Calculate loss (simplified)
                loss = self.criterion(class_pred.mean(dim=[2, 3]), labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(class_pred.mean(dim=[2, 3]), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = epoch_loss / len(self.trainloader)
            accuracy = correct / total

            print(f"   Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")
            total_loss += avg_loss

        return {
            "train_loss": total_loss / epochs,
            "train_accuracy": accuracy
        }

    def evaluate_model(self) -> Tuple[float, float, Dict]:
        """Evaluate face detection model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, bboxes, labels in self.valloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                bbox_pred, class_pred = self.model(images)

                loss = self.criterion(class_pred.mean(dim=[2, 3]), labels)
                total_loss += loss.item()

                _, predicted = torch.max(class_pred.mean(dim=[2, 3]), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.valloader)
        accuracy = correct / total

        return avg_loss, accuracy, {"val_loss": avg_loss}

    def get_model_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def _get_dataset_size(self) -> int:
        """Return training dataset size."""
        return len(self.trainloader.dataset)