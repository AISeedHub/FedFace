# Federated Learning for Face Detection using PyTorch and Flower

## Overview 
```aiignore
┌─────────────────────────────────────────────────────────────┐
│                     Multi-Repo Setup                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  fedflower-core (PyPI Package)                               │
│  └─→ Provides: Server, Client, Strategies                   │
│       │                                                       │
│       │ pip install fedflower-core                           │
│       │                                                       │
│       ▼                                                       │
│  fed-face-detection                                    │
│  └─→ Uses: fedflower-core                                    │
│  └─→ Provides: Face models, training scripts                │
│                                                               │
└─────────────────────────────────────────────────────────────┘

```

### 1. Dependency Chain:

```aiignore
fedflower-face-detection
    ├── requirements.txt
    │   └── fedflower-core>=1.0.0  ← Install from https://github.com/AISeedHub/FedFlower
    │
    └── src/federated/face_client.py
        └── from fedflower.client import FedFlowerClient  ← Import from core
```
### 2. Interface Contract:
`fedflower-core` defines abstract base class FedFlowerClient
`fedflower-face-detection` implements task-specific methods
Server uses strategy pattern from core

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Organization                       │
│                      AISeedHub/                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │ fedflower-core   │◄───│ fedflower-face-detection    │   │
│  │  (Framework)     │    │   (Face Detection App)      │   │
│  │                  │    │                              │   │
│  │  • Server        │    │  • MobileNetV3-SSD          │   │
│  │  • Client Base   │    │  • WIDER FACE Dataset       │   │
│  │  • Strategies    │    │  • Mobile optimization      │   │
│  └──────────────────┘    └─────────────────────────────┘   │
│         ▲                            │                       │
│         │                            │                       │
│         │         pip install        │                       │
│         └────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

                         Deploy to:
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   [PC Server]          [PC Desktop]        [Smartphone 5]
  Run FL Server        Run FL Client         Run FL Client
  
```

## Workflow
```aiignore
Round 1:
┌─────────────┐
│   Server    │  1. Broadcast initial model
│  (PC/Cloud) │────────────────────────┐
└─────────────┘                        │
                                       ▼
                        ┌──────────────────────────┐
                        │   Client 0 (Phone 1)     │
                        │   • Load local data      │
                        │   • Train 2 epochs       │
                        │   • Compute gradients    │
                        └──────────────────────────┘
                                       │
                                       │ 2. Send updates
                                       ▼
┌─────────────┐                ┌──────────┐
│   Server    │ 3. Aggregate   │ Updates  │
│  FedAvg     │◄───────────────┤ from all │
│             │    (FedAvg)    │ clients  │
└─────────────┘                └──────────┘
       │
       │ 4. Broadcast updated model
       ▼
  (Next Round...)
```

## Run End-to-End Demo 🎬

Terminal 1: Start Server (PC)
```bash
cd fedflower-face-detection

python train.py \
  --mode server \
  --config configs/mobile_5clients.yaml
  
  ```
Expected Output:
```
🖥️  Starting Face Detection FL Server
🌸 Starting FedFlower Server on 0.0.0.0:9000
📊 Rounds: 20 | Min Clients: 3

INFO flwr 2025-10-27 09:16:45 | app.py:163 | Starting Flower server, config: num_rounds=20, no SSL
INFO flwr 2025-10-27 09:16:45 | server.py:89 | Flower ECE: gRPC server running (20 rounds), SSL is disabled
INFO flwr 2025-10-27 09:16:45 | server.py:89 | [INIT]
INFO flwr 2025-10-27 09:16:45 | server.py:89 | Requesting initial parameters from one random client
```

Terminal 2-6: Start 5 Clients (Smartphones or PCs)
Client 0:

```bash
# On PC
cd fedface/src

python train.py \
  --mode client \
  --client-id 0 \
  --config configs/mobile_5clients.yaml \
  --server-address 192.168.1.100:9000
  
  ```

Client 1:

```bash
python train.py --mode client --client-id 1 --server-address 192.168.1.100:9000
... (repeat for clients 2, 3, 4)
```

Expected Client Output:
```
Code
📱 Starting Face Detection Client 0
📱 Client 0 initialized
   Model size: 8.42 MB
   Dataset size: 2000 images

INFO flwr 2025-10-27 09:17:01 | grpc.py:52 | Opened insecure gRPC connection (no certificates were passed)
INFO flwr 2025-10-27 09:17:02 | connection.py:42 | ChannelConnectivity.READY

[Client 0] Starting training round...
   Epoch 1/2: loss=0.6234, acc=0.7123
   Epoch 2/2: loss=0.5456, acc=0.7589

[Client 0] Evaluating...
   Validation: loss=0.5123, acc=0.7834
   
   ```

Server Output During Training:
```
INFO flwr 2025-10-27 09:17:05 | server.py:89 | FL starting
DEBUG flwr 2025-10-27 09:17:05 | server.py:222 | fit_round 1: strategy sampled 5 clients (out of 5)

INFO flwr 2025-10-27 09:18:23 | server.py:125 | fit_round 1 received 5 results and 0 failures
DEBUG flwr 2025-10-27 09:18:23 | server.py:173 | evaluate_round 1: strategy sampled 5 clients

INFO flwr 2025-10-27 09:18:45 | server.py:148 | evaluate_round 1 received 5 results and 0 failures
INFO flwr 2025-10-27 09:18:45 | server.py:222 | 
	[ROUND 1]
	loss: 0.5421
	accuracy: 0.7456
	distributed_fit_time: 78.2s
	distributed_evaluate_time: 22.1s

... (continues for 20 rounds)
```
