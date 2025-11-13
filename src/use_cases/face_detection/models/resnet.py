import torch

# import resnet from torchvision.models
from torchvision.models import ResNet18_Weights, resnet18

from src.use_cases.face_detection.models import FaceDetectorBase


class PretrainedResNet(FaceDetectorBase):
    """
    Simple ResNet18 for image classification.
    Used for demonstration purposes.

    Input: 3x224x224 images
    Output: num_classes logits
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer to match num_classes
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Conv layers
        x = self.resnet(x)

        return x
