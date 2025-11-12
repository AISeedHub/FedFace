
"""Base model interface for face detection"""

from abc import ABC, abstractmethod
import torch.nn as nn


class FaceDetectorBase(nn.Module, ABC):
    """
    Base class for all face detection models.
    All models should inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass

    def get_num_parameters(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())