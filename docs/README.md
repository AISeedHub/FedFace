# Tutorial: FedFace

This `FedFace` project leverages **Federated Learning** for *face classification*. It allows multiple clients to *collaboratively train* an AI model on their *local, private face data* without ever sending the raw data to a central server. The *Federated Server* coordinates this learning, aggregating model updates, while a *configuration system* dictates all operational parameters. The architecture is modular, supporting *pluggable AI models* and robust *data management* for various distribution scenarios.


## Visual Overview

![image](../assets/visual_overview.png)

## Chapters

1. [Configuration System
](01_configuration_system_.html)
2. [Federated Server (FedFlowerServer)
](02_federated_server__fedflowerserver__.html)
3. [Federated Client (FedFlowerClient)
](03_federated_client__fedflowerclient__.html)
4. [Data Management and Distribution
](04_data_management_and_distribution_.html)
5. [Face Classification Logic
](05_face_classification_logic_.html)
6. [Pluggable Model Architecture
](06_pluggable_model_architecture_.html)

---
