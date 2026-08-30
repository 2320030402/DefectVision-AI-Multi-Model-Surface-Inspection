"""
Model architecture definitions — must stay IDENTICAL to the ones used in the
training notebook (Capstone1_Defect_Detection_Skeleton.ipynb), since a checkpoint
saved from the notebook is only loadable into a matching architecture here.
"""
import torch.nn as nn
from torchvision import models

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted",
    "rolled",
    "scratches",
]  # confirmed from the training notebook's classification report — alphabetical
   # order matches Section 4's sorted() folder listing, so index positions
   # (0-5) are unchanged from earlier; only the display names were corrected.
NUM_CLASSES = len(CLASSES)


def build_resnet18(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)  # weights loaded from checkpoint, not ImageNet, at inference time
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenetv2(num_classes=NUM_CLASSES):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


MODEL_REGISTRY = {
    "resnet18": build_resnet18,
    "mobilenet_v2": build_mobilenetv2,
    "custom_cnn": CustomCNN,
}

# Which layer to target for Grad-CAM, per architecture
GRADCAM_TARGET_LAYER = {
    "resnet18": lambda m: m.layer4[-1],
    "mobilenet_v2": lambda m: m.features[-1],
    "custom_cnn": lambda m: m.features[-4],
}
