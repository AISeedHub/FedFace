# A User Guide for Face Recognition with FedFlower

This guide provides instructions on how to use the face recognition use case in the FedFlower framework. It covers data preparation, configuration settings, and running the face recognition tasks.

## Folder Structure

The folder structure for the face recognition use case is as follows:

```
src/use_cases/
└── face_detection/
    ├── configs/
    │   └── base.yaml          # Configuration file
    ├── utils/
    │   ├── distribute_data.py # Utility functions for data
    │   └── lfw_100_loader.py  # Data loader for LFW dataset
    ├── models/
    │   ├── cnn.py      # CNN model definition
    |   ├── mlp.py      # MLP model definition
    │   └── resnet.py   # ResNet model definition
    ├── data/              # Contains dataset (original) and Data processing if needed: loading, augmentation,...
    │   └──data.npz        # Sample dataset file   
    ├── main_client.py     # Main client script
    ├── main_server.py     # Main server script
    ├── central_run.py     # Centralized training script
    ├── distributed_data/  # Directory for distributed data
    └── README.md          # This user guide
```

# Environment Setup

To set up the environment for running the face recognition use case, follow these steps:

1. **Clone the Repository**: Ensure you have cloned the repository to your local machine.
2. **Install Dependencies**: Navigate to the root directory of the repository and install the required dependencies using uv:

```bash
  uv sync
```

notes: Make sure you have `uv` installed on your system. If not, you can install it via pip:

```bash
  pip install uv
```

# Configuration

The configuration for the face recognition use case is managed through the `base.yaml` file located in the `configs` directory. Key parameters include:

- `server_address`: The address of the server.
- `num_rounds`: The number of training rounds.
- `min_clients`: The minimum number of clients required for training.
- `data_type`: The format of the data, either "folder" or "npz".
- `full_data_path`: The path to the full dataset.
- `distributed_data_path`: The path where the distributed data will be stored.
- `num_clients`: The number of clients to distribute the data among.
- `non_iid`: A boolean indicating whether to use non-IID data distribution.
- `alpha`: The Dirichlet distribution parameter for non-IID data splitting.
- `model`: The model architecture to be used (e.g., "cnn", "mlp", "resnet").
- `learning_rate`: The learning rate for training.
- `batch_size`: The batch size for training.
- `num_classes`: The number of classes in the dataset.
- `local_epochs`: The number of local epochs for each client.
- `batch_size`: The batch size for training.

# Running the Use Case

1. Distribute the Data:
   Run the `distribute_data.py` script to distribute the dataset among the specified number of clients based on the configuration settings.

```bash  
python src/use_cases/face_detection/utils/distribute_data.py
```

| Scenario          | Output Data                                      | Notes                          |
|-------------------|----------------------------------------------------------------|--------------------------------|
| Using npz data     | data.npz (numpy array) | Ensure `data_type` is set to "npz" in `base.yaml` |
| Using folder data  | folder structure with images | Ensure `data_type` is set to "folder" in `base.yaml` |

1. Start the Server:
   Run the `main_server.py` script to start the server.

```bash  
python src/use_cases/face_detection/main_server.py
```

3. Start the Clients:
   For each client, run the `main_client.py` script to start the client processes.

```bash
python src/use_cases/face_detection/main_client.py --client-id [id]
```

4. Centralized Training (Optional):
   If you want to run centralized training, use the `central_run.py` script.

```bash
python src/use_cases/face_detection/central_run.py
```
