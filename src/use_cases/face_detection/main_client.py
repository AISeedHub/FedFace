"""Main client for face classification federated learning"""

import yaml
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.fed_core.fed_client import FedFlowerClient
from src.use_cases.face_detection.models.cnn import SimpleCNN


class FaceClassificationClient(FedFlowerClient):
    """Face Classification Client for Federated Learning"""
    
    def __init__(self, client_id: int, config: Dict):
        super().__init__(client_id, config)
        
        # Initialize model
        self.model = SimpleCNN(num_classes=config['model']['num_classes'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=config['learning_rate'],
            momentum=0.9
        )
        
        # Load client data
        self.train_loader, self.test_loader = self._load_data()
        
        print(f"[Client {client_id}] Initialized with {len(self.train_loader.dataset)} training samples")
    
    def _load_data(self):
        """Load client-specific data"""
        data_path = os.path.join(
            self.config['data_path'], 
            f"client_{self.client_id}"
        )
        
        # Load images and labels
        images_path = os.path.join(data_path, "images.pt")
        labels_path = os.path.join(data_path, "labels.pt")
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Data not found for client {self.client_id}. "
                f"Please run data distribution first."
            )
        
        images = torch.load(images_path)
        labels = torch.load(labels_path)
        
        # Resize images to match model input (32x32)
        if images.shape[-1] != 32:
            images = torch.nn.functional.interpolate(
                images, size=(32, 32), mode='bilinear', align_corners=False
            )
        
        # Split into train/test (80/20)
        dataset_size = len(images)
        train_size = int(0.8 * dataset_size)
        
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        test_images = images[train_size:]
        test_labels = labels[train_size:]
        
        # Create datasets and loaders
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_model(self, epochs: int) -> Dict[str, float]:
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
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / (epochs * len(self.train_loader))
        
        print(f"[Client {self.client_id}] Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy
        }
    
    def evaluate_model(self) -> Tuple[float, float, Dict]:
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
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy, {"test_accuracy": accuracy}
    
    def get_model_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def _get_dataset_size(self) -> int:
        """Return size of local training dataset"""
        return len(self.train_loader.dataset)


def load_config(config_path="src/use_cases/face_detection/configs/base.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def main():
    """Start federated learning client"""
    parser = argparse.ArgumentParser(description="Face Classification Federated Client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (0, 1, ...)")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:9000", help="Server address")
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
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()
