"""Lightweight Face Detection Model for Mobile"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3_SSD(nn.Module):
    """
    MobileNetV3 + SSD for Face Detection
    Optimized for mobile deployment
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Backbone: MobileNetV3-Small
        self.backbone = mobilenet_v3_small(pretrained=pretrained)

        # Remove classifier
        self.features = self.backbone.features

        # SSD detection heads (simplified)
        self.detection_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * 4, kernel_size=3, padding=1),  # 4 = bbox coords
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)

        # Detection
        bbox_pred = self.detection_head(features)
        class_pred = self.classification_head(features)

        return bbox_pred, class_pred

    def get_model_size_mb(self):
        """Calculate model size in MB"""
        param_size = sum(p.numel() for p in self.parameters()) * 4 / (1024 ** 2)
        return param_size