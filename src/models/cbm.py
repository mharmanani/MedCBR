import torch
from torch import nn
from torchvision import models


def _make_backbone(style: str) -> nn.Module:
    if style == "resnet18":
        model = models.resnet18(weights="DEFAULT")
    elif style == "resnet50":
        model = models.resnet50(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported backbone '{style}'. Use 'resnet18' or 'resnet50'.")

    model.fc = nn.Identity()
    return model


def _get_embedding_dims(style: str) -> int:
    return {
        "resnet18": 512,
        "resnet50": 2048,
    }[style]


class BaseCBM(nn.Module):
    """Concept Bottleneck Model with a simple feedforward concept head and class head, and a ResNet backbone.
    Follows Koh et al. (2020) "Concept Bottleneck Models" (https://arxiv.org/abs/2007.04612)
    """
    SUPPORTED_BACKBONES = {"resnet18", "resnet50"}

    def __init__(
        self,
        num_concepts: int = 5,
        num_classes: int = 5,
        backbone: str = "resnet50",
    ):
        super(BaseCBM, self).__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Use one of {sorted(self.SUPPORTED_BACKBONES)}"
            )

        self.backbone = _make_backbone(backbone)
        embedding_dim = _get_embedding_dims(backbone)

        self.concept_layer = nn.Linear(embedding_dim, num_concepts)
        self.clf_layer = nn.Linear(num_concepts, num_classes)
        self.relu = nn.ReLU()

    def load_backbone_weights(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not ("clf_layer" in k or "concept_layer" in k)
        }
        self.backbone.load_state_dict(state_dict)

    def extract_concepts(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        c = self.concept_layer(x)
        return c

    def forward(self, x):
        c = self.extract_concepts(x)
        c = torch.sigmoid(c)
        y = self.clf_layer(c)
        return c, y