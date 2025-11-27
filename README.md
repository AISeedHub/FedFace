# Federated Learning for Face Classification using PyTorch and Flower

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/873bafde-94f2-4e6a-b964-61b45d4ed2da" />


## Overview 
### Chasing `Pluggable Models` + `Config-Driven Design` + `Modular Architecture` concept

```aiignore
src/
├── fed_core/                  # 1. Federated Learning Core
│   ├── fed_client.py              # Common logic for client (training, update model)
│   └── fed_server.py              # Common logic for server (aggregate, distribute model)
│
├── use_cases/                 # 2. Specific application use cases
│   └── face_detection/       
│       │
│       ├── configs/           # 3. Configs Directory - Very important!
│       │   └── base.yaml
│       │
│       ├── models/            # 4. "Pluggable" AI Models Architecture
│       │   ├── __init__.py    # Contains "model factory" to select model, Interface (base class) for all models
│       │   ├── mlp/
│       │   │   ├── __init__.py
│       │   │   └── architecture.py
│       │   └── cnn/
│       │       ├── __init__.py
│       │       └── architecture.py
│       │
│       ├── data/              # Contains dataset (original) and Data processing if needed: loading, augmentation,...
│       │   ├──data.npz        # Sample dataset file   
│       │   └── ...
│       │
│       ├── utils/
│       │   ├── distribute_data.py # Data distribution utility
│       │   └── prepare_dataset.py # Dataset preparation
│       │
│       ├── distributed_data/      # Client data storage
│       │   ├── client_0/
│       │   ├── client_1/
│       │   └── ...
│       │
│       ├── central_run.py   # Script to run centralized model (non-federated) for comparison
│       │
│       ├── main_server.py     # 5. Entry point to run Server
│       └── main_client.py     # 6. Entry point to run Client
│    
├── run_clients.sh    # Script to launch multiple clients
├── run_server.sh     # Script to launch server
├── run_central.sh    # Script to launch centralized training
│
├── pyproject.toml            # Project configuration
├── uv.lock                   # Dependency lock file
└── README.md
```

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

## Quick Start
1. Setup Environment, [check out ](src/use_cases/face_detection/README.md:30)
2. Start Server
`./src/run_server.bat` (for Windows)
3. Start Clients
`./src/run_client.bat <client_id>` (for Windows)

## Config and Usage


### 1.Configuration

Edit `src/use_cases/face_detection/configs/base.yaml` to customize:

```yaml
# Server Configuration
server_address: "0.0.0.0:9000" # public server address
num_rounds: 5
min_clients: 2 # minimum clients to start training

# Training Configuration
local_epochs: 3
batch_size: 32
learning_rate: 0.01

# Model Configuration
model:
  name: "resnet"
  num_classes: 100

# Data Configuration
data_path: "src/use_cases/face_detection/distributed_data"
num_clients: 2 # number of clients
```

### 2. Start the Server
- First, generate and distribute synthetic data for `num_clients` clients
- Then, start the server


In terminal:
+ On Windows:
    ```bash
    ./src/run_server.bat
    ```
  + If on Linux and GitBash:
      ```bash
      bash ./src/run_server.sh
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
.
.
.
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

### 3. Start Client 

- In a second terminal:
    + On Windows:
        ```bash
        ./src/run_clients.bat 0
        ```
      + If on Linux and GitBash:
          ```bash
          bash ./src/run_clients.sh 0
          ```

- In a third terminal:
  + On Windows:
    ```bash
    ./src/run_clients.bat 1
    ```
    + If on Linux and GitBash:
        ```bash
        bash ./src/run_clients.sh 1
        ```
- ETC for more clients if any

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
...
```



#### Copyright © 2025 AISEED. All rights reserved.
