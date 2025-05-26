import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)


class MobilenetV2(nn.Module):
    def __init__(self, cnn_Embed_out=640, Classif_kernel=1):
        super(MobilenetV2, self).__init__()

        mobilenetv2 = models.mobilenet_v2(pretrained=True)
        layers = list(mobilenetv2.features.named_children())[:-1]
        self.base = nn.Sequential(OrderedDict(layers))
        self.norm = L2N()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cnn_Embed = nn.Sequential(
            nn.Conv2d(320, cnn_Embed_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_Embed_out),
            nn.ReLU6(inplace=True)
        )
        self.fc_Embed = nn.Sequential(
            nn.Linear(cnn_Embed_out, 480),
            nn.Linear(480, 320)
        )

        self.cnn_Classif = nn.Sequential(
            nn.Conv2d(320, 480, kernel_size=Classif_kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(480, 160, kernel_size=1, padding=1),
            nn.ReLU(),
        )
        self.fc_Classif = nn.Sequential(
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 2)
        )

    def forward(self, x, mode=None):
        x = self.base(x)
        embeddings = self.cnn_Embed(x)
        embeddings = self.pool(embeddings).squeeze(-1).squeeze(-1)
        embeddings = self.fc_Embed(embeddings)
        embeddings = self.norm(embeddings)

        if mode == "embedding_only":
            return embeddings

        class_logits = self.cnn_Classif(x)
        class_logits = self.pool(class_logits).squeeze(-1).squeeze(-1)
        class_logits = self.fc_Classif(class_logits)

        return embeddings, class_logits