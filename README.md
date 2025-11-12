# Federated Learning for Face Classification using PyTorch and Flower

## Overview 
### Chasing `Pluggable Models` + `Config-Driven Design` + `Modular Architecture` 

```aiignore
├── fed_core/                  # 1. Lõi Federated Learning
│   ├── client.py              # Logic chung cho client (training, update model)
│   ├── server.py              # Logic chung cho server (aggregate, distribute model)
│   ├── strategy/              # Các chiến lược tổng hợp (FedAvg, FedProx,...)
│   │   ├── __init__.py
│   │   ├── fed_avg.py
│   │   └── base_strategy.py
│   └── communication/         # Giao thức giao tiếp client-server
│       └── grpc_comm.py       # (hoặc các phương thức khác)
│
├── use_cases/                 # 2. Các bài toán ứng dụng cụ thể
│   └── face_detection/        # Bài toán Face Detection (trước đây là FedFace)
│       │
│       ├── configs/           # 3. Thư mục Configs - Rất quan trọng!
│       │   ├── base_config.yaml
│       │   ├── retinaface_pascal_voc.yaml  # Config cho model RetinaFace
│       │   └── ssd_widerface.yaml          # Config cho model SSD
│       │
│       ├── models/            # 4. Kiến trúc "Pluggable" AI Models
│       │   ├── __init__.py    # Chứa "model factory" để chọn model
│       │   ├── base_model.py  # Interface (lớp cơ sở) cho mọi model
│       │   ├── ssd/
│       │   │   ├── __init__.py
│       │   │   └── architecture.py
│       │   └── retinaface/
│       │       ├── __init__.py
│       │       └── architecture.py
│       │
│       ├── data/              # Xử lý data cho face detection
│       │   ├── widerface_loader.py
│       │   ├── pascal_voc_loader.py
        │   └── distribute_data.py            # Script để tạo và chia dữ liệu
│       │
│       ├── main_server.py     # 5. Entry point để chạy Server
│       └── main_client.py     # 6. Entry point để chạy Client
│
├── requirements.txt           # Thư viện chung
└── README.md

```

This implementation provides a complete federated learning system for face classification using the Flower framework with 1 server and 2 clients.

## Overview

- **Server**: Coordinates federated learning across multiple clients using FedAvg strategy
- **Clients**: Train a SimpleCNN model locally on distributed face classification data
- **Model**: SimpleCNN with 10 classes for face classification
- **Data**: Synthetic face-like data distributed in Non-IID fashion (80-20 split)

## Architecture

```
┌─────────────────┐
│   Fed Server    │ ← Coordinates training, aggregates models
│   (Port 9000)   │
└─────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Client0│ │Client1│ ← Train locally on distributed data
│(800)  │ │(200)  │
└───────┘ └───────┘
```

## Setup and Usage

### 1. Prepare Data

First, generate and distribute synthetic data for 2 clients:

```bash
cd src/use_cases/face_detection/utils
python distribute_data.py --num-clients 2 --num-images 1000 --non-iid
```

This creates:
- Client 0: 800 images (80%)
- Client 1: 200 images (20%)
- Non-IID distribution for realistic federated learning scenario

### 2. Start the Server

In one terminal:

```bash
python src/use_cases/face_detection/main_server.py
```

Expected output:
```
🌸 FedFlower - Face Classification Server
==================================================
🚀 Starting server with 2 clients
📊 Training rounds: 5
🎯 Model: simple_cnn (10 classes)
==================================================
🌸 Starting FedFlower Server on 0.0.0.0:9000
📊 Rounds: 5 | Min Clients: 2
```

### 3. Start Client 0

In a second terminal:

```bash
python src/use_cases/face_detection/main_client.py --client-id 0
```

### 4. Start Client 1

In a third terminal:

```bash
python src/use_cases/face_detection/main_client.py --client-id 1
```

Expected client output:
```
🌸 FedFlower - Face Classification Client 0
==================================================
[Client 0] Initialized with 640 training samples
🚀 Connecting to server at 127.0.0.1:9000
==================================================
```

## Configuration

Edit `src/use_cases/face_detection/configs/base.yaml` to customize:

```yaml
# Server Configuration
server_address: "0.0.0.0:9000"
num_rounds: 5
min_clients: 2

# Training Configuration
local_epochs: 3
batch_size: 32
learning_rate: 0.01

# Model Configuration
model:
  name: "simple_cnn"
  num_classes: 10
```

## Testing

Run the test script to verify the implementation:

```bash
python test_face_classification.py
```

## Project Structure

```
src/use_cases/face_detection/
├── main_server.py          # Federated server entry point
├── main_client.py          # Federated client implementation
├── configs/
│   └── base.yaml          # Configuration file
├── models/
│   ├── __init__.py        # Base model interface
│   └── cnn.py            # SimpleCNN model
├── utils/
│   ├── distribute_data.py # Data distribution utility
│   └── prepare_dataset.py # Dataset preparation
└── distributed_data/      # Client data storage
    ├── client_0/
    │   ├── images.pt
    │   └── labels.pt
    └── client_1/
        ├── images.pt
        └── labels.pt
```

## Expecting Results

The system successfully trains a face classification model across 2 clients:
- **Client 0**: 800 samples → ~50% accuracy
- **Client 1**: 200 samples → ~45% accuracy
- **Federated Model**: Aggregated model from both clients


