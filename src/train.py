"""
Main Training Script for Federated Face Detection

Usage:
  Server: python train.py --mode server --config configs/mobile_5clients.yaml
  Client: python train.py --mode client --client-id 0 --config configs/mobile_5clients.yaml
"""

import argparse
import yaml
import torch
import flwr as fl

# Import from fedflower-core
from fedflower.server import FedFlowerServer

# Import from this repo
from src.federated_client import FaceDetectionClient
from src.data.wider_face_loader import prepare_federated_data


def load_config(config_path: str):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_server(config: dict):
    """Run Federated Learning Server."""
    print("🖥️  Starting Face Detection FL Server")

    server = FedFlowerServer(
        num_rounds=config['server']['num_rounds'],
        min_clients=config['server']['min_clients'],
        config=config['training']
    )

    server.start(server_address=config['server']['address'])


def run_client(client_id: int, config: dict, server_address: str):
    """Run Federated Learning Client."""
    print(f"📱 Starting Face Detection Client {client_id}")

    # Prepare data
    trainloaders, valloaders, testloader = prepare_federated_data(
        num_clients=config['num_clients'],
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size']
    )

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create client
    client = FaceDetectionClient(
        client_id=client_id,
        trainloader=trainloaders[client_id],
        valloader=valloaders[client_id],
        config=config,
        device=device
    )

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


def main():
    parser = argparse.ArgumentParser(description="Federated Face Detection")
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client"])
    parser.add_argument("--config", type=str, default="configs/mobile_5clients.yaml")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--server-address", type=str, default=None)

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.mode == "server":
        run_server(config)
    else:
        server_addr = args.server_address or config['server']['address']
        run_client(args.client_id, config, server_addr)


if __name__ == "__main__":
    main()