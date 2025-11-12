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

Expected output:
```
Andrew@DESKTOP-3D6VN4M MINGW64 ~/Documents/Project/FedFace/src/use_cases/face_detection/utils (main)
$ uv run python distribute_data.py --num-clients 2 --num-images 1000 --non-iid
================================================================================
🌸 FedFlower - Face Detection Data Preparation
================================================================================

📥 Step 1: Preparing dataset...

⚠️  Using synthetic data (set use_real_data=True for real faces)

🎨 Creating synthetic face detection data...
   - Images: 1000
   - Size: 64x64x3


📤 Step 2: Distributing to 2 clients...

================================================================================
📊 Distributing 1000 images to 2 clients
   Distribution: Non-IID
================================================================================

   Split ratio: 80% - 20%

--------------------------------------------------------------------------------
✓ Client 0:
   - Images:  800 (80.0%)
   - Label distribution: {0: 380, 1: 420}
   - Saved to: use_cases/face_detection/distributed_data\client_0
✓ Client 1:
   - Images:  200 (20.0%)
   - Label distribution: {0: 104, 1: 96}
   - Saved to: use_cases/face_detection/distributed_data\client_1
--------------------------------------------------------------------------------

✅ Data distribution complete!
📁 Data saved to: use_cases/face_detection/distributed_data


================================================================================
✅ All done! You can now start the federated learning process.
================================================================================

Next steps:
  1. Start server: python use_cases/face_detection/main_server.py
  2. Start clients: python use_cases/face_detection/main_client.py --client-id 0
================================================================================
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
Andrew@DESKTOP-3D6VN4M MINGW64 ~/Documents/Project/FedFace (main)
$ uv run python src/use_cases/face_detection/main_server.py
🌸 FedFlower - Face Classification Server
==================================================
🚀 Starting server with 2 clients
📊 Training rounds: 5
🎯 Model: simple_cnn (10 classes)
==================================================
🌸 Starting FedFlower Server on 0.0.0.0:9000
📊 Rounds: 5 | Min Clients: 2
WARNING :   DEPRECATED FEATURE: flwr.server.start_server() is deprecated.
        Instead, use the `flower-superlink` CLI command to start a SuperLink as shown below:

                $ flower-superlink --insecure

        To view usage and all available options, run:

                $ flower-superlink --help

        Using `start_server()` is deprecated.

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.

INFO :      Starting Flower server, config: num_rounds=5, no round_timeout
INFO :      Flower ECE: gRPC server running (5 rounds), SSL is disabled
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
INFO :      Evaluation returned no results (`None`)
INFO :
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
INFO :      aggregate_fit: received 2 results and 0 failures
WARNING :   No fit_metrics_aggregation_fn provided
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :
INFO :      [ROUND 4]
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :
INFO :      [ROUND 5]
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :
INFO :      [SUMMARY]
INFO :      Run finished 5 round(s) in 20.86s
INFO :          History (loss, distributed):
INFO :                  round 1: 0.7038475275039673
INFO :                  round 2: 0.7127608776092529
INFO :                  round 3: 0.6975250959396362
INFO :                  round 4: 0.700778579711914
INFO :                  round 5: 0.69382164478302
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 47.5), (2, 47.5), (3, 52.5), (4, 52.5), (5, 52.5)]}
INFO :
(base)
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
Andrew@DESKTOP-3D6VN4M MINGW64 ~/Documents/Project/FedFace (main)
$ uv run python src/use_cases/face_detection/main_client.py --client-id 0
🌸 FedFlower - Face Classification Client 0
==================================================
[Client 0] Initialized with 640 training samples
🚀 Connecting to server at 127.0.0.1:9000
==================================================
INFO :
INFO :      Received: get_parameters message a3eaf6ed-06f5-4eb0-94cc-04ccca828e28
INFO :      Sent reply
INFO :
INFO :      Received: train message 5758d188-f4fb-4dce-981c-3e141a89203d
[Client 0] Starting training round...
[Client 0] Epoch 1/3, Loss: 1.7948
[Client 0] Epoch 2/3, Loss: 1.0733
[Client 0] Epoch 3/3, Loss: 0.7040
[Client 0] Training completed - Loss: 1.1907, Accuracy: 44.84%
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 8665a821-d4e6-49b8-b6ea-c66fc64896bc
[Client 0] Evaluating...
[Client 0] Evaluation - Loss: 0.7042, Accuracy: 47.50%
INFO :      Sent reply
INFO :
INFO :      Received: train message bbc1548e-888e-4f05-bb66-2c02dcecd2f3
[Client 0] Starting training round...
[Client 0] Epoch 1/3, Loss: 0.7230
[Client 0] Epoch 2/3, Loss: 0.7223
[Client 0] Epoch 3/3, Loss: 0.7021
[Client 0] Training completed - Loss: 0.7158, Accuracy: 50.62%
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message fe239abc-1a37-41fd-92f4-4bf3cc94e58c
[Client 0] Evaluating...
[Client 0] Evaluation - Loss: 0.7133, Accuracy: 47.50%
INFO :      Sent reply
INFO :
INFO :      Received: train message d1424030-f7ba-443f-93c5-42a384b40665
[Client 0] Starting training round...
[Client 0] Epoch 1/3, Loss: 0.7044
[Client 0] Epoch 2/3, Loss: 0.7041
[Client 0] Epoch 3/3, Loss: 0.7105
[Client 0] Training completed - Loss: 0.7063, Accuracy: 49.17%
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 5d0e1a9d-aa26-408a-89fd-b05dad11fae3
[Client 0] Evaluating...
[Client 0] Evaluation - Loss: 0.6973, Accuracy: 52.50%
INFO :      Sent reply
INFO :
INFO :      Received: train message 82be0df8-b16c-46c2-afd8-65c11d7efe63
[Client 0] Starting training round...
[Client 0] Epoch 1/3, Loss: 0.7140
[Client 0] Epoch 2/3, Loss: 0.7013
[Client 0] Epoch 3/3, Loss: 0.7074
[Client 0] Training completed - Loss: 0.7076, Accuracy: 48.65%
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 666ef843-6335-48b1-b54b-3075ea5e6fd4
[Client 0] Evaluating...
[Client 0] Evaluation - Loss: 0.7002, Accuracy: 52.50%
INFO :      Sent reply
INFO :
INFO :      Received: train message ab1f1b38-0139-41cd-a26f-355e0259d289
[Client 0] Starting training round...
[Client 0] Epoch 1/3, Loss: 0.7127
[Client 0] Epoch 2/3, Loss: 0.6985
[Client 0] Epoch 3/3, Loss: 0.7025
[Client 0] Training completed - Loss: 0.7045, Accuracy: 48.85%
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 9bb559b6-e37d-4036-be8d-9700e3f091db
[Client 0] Evaluating...
[Client 0] Evaluation - Loss: 0.6935, Accuracy: 52.50%
INFO :      Sent reply
INFO :
INFO :      Received: reconnect message 196409d5-e523-4b3b-9116-60ca5fafcf7b
INFO :      Disconnect and shut down
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


