import torch
import torch.nn as nn

def to_one_hot(vec, num_classes):
    """
    Convert a vector of integer labels into a one-hot encoded matrix.
    
    Args:
    - vec (torch.Tensor): A 1D tensor of shape (N,) with integer values.
    - num_classes (int): The number of classes (output dimension for one-hot encoding).
    
    Returns:
    - torch.Tensor: One-hot encoded tensor of shape (N, num_classes).
    """
    return torch.eye(num_classes)[vec].unsqueeze(1)

def from_one_hot(one_hot_vec):
    """
    Convert a one-hot encoded matrix back to a vector of integer labels.
    
    Args:
    - one_hot_vec (torch.Tensor): A 2D tensor of shape (N, num_classes) where each row is a one-hot vector.
    
    Returns:
    - torch.Tensor: A 1D tensor of shape (N,) with integer class labels.
    """
    return torch.argmax(one_hot_vec, dim=1)

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
