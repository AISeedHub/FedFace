import torch.nn as nn

from src.use_cases.face_detection.models import FaceDetectorBase


class MLP(FaceDetectorBase):
    """
    Simple MLP for classification head.
    Used for demonstration purposes.

    Input: embedding vectors of size 512
    Output: num_classes logits
    """

    def __init__(self, num_classes=10):
        super(MLP, self).__init__()

        # Fully connected layers
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # FC layers
        x = self.mlp(x)

        return x
