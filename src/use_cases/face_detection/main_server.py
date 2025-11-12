"""Main server for face classification federated learning"""

import yaml
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.fed_core.fed_server import FedFlowerServer


def load_config(config_path="src/use_cases/face_detection/configs/base.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def main():
    """Start the federated learning server for face classification"""
    print("🌸 FedFlower - Face Classification Server")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Create server
    server = FedFlowerServer(
        num_rounds=config['num_rounds'],
        min_clients=config['min_clients'],
        config=config
    )
    
    # Start server
    print(f"🚀 Starting server with {config['num_clients']} clients")
    print(f"📊 Training rounds: {config['num_rounds']}")
    print(f"🎯 Model: {config['model']['name']} ({config['model']['num_classes']} classes)")
    print("=" * 50)
    
    server.start(config['server_address'])


if __name__ == "__main__":
    main()
