import torch.nn as nn
import torchvision.models as models

class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()
        self.model = models.resnet18(weights = "IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)