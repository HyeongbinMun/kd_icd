# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import enum
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.regnet import regnet_x_800mf, regnet_y_800mf
from torchvision.models.mobilenet import mobilenet_v2
from classy_vision.models import build_model
from .gem_pooling import GlobalGeMPool2d
from .mobilenetv2 import MobilenetV2
from pytorch_lightning import LightningModule


class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)


class Implementation(enum.Enum):
    CLASSY_VISION = enum.auto()
    TORCHVISION = enum.auto()
    TIMM = enum.auto()
    VIT = enum.auto()

class Backbone(enum.Enum):
    CV_RESNET18 = ("resnet18", 512, Implementation.CLASSY_VISION)
    CV_RESNET50 = ("resnet50", 2048, Implementation.CLASSY_VISION)
    CV_RESNEXT101 = ("resnext101_32x4d", 2048, Implementation.CLASSY_VISION)

    TV_EFFICIENTNET_B0 = (efficientnet_b0, 1280, Implementation.TORCHVISION)
    TV_MOBILENETV2 = (mobilenet_v2, 320, Implementation.TORCHVISION)
    TV_MOBILENETV3 = (mobilenet_v3_large, 1280, Implementation.TORCHVISION) # 
    TV_REGNET_X = (regnet_x_800mf, 672, Implementation.TORCHVISION)
    TV_REGNET_Y = (regnet_y_800mf, 440, Implementation.TORCHVISION) # 6,432,512 sum(p.numel() for p in models.regnet.regnet_y_800mf().parameters())

    TV_RESNET18 = (resnet18, 512, Implementation.TORCHVISION)
    TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
 #   TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)

    TIMM_MOBILEVIT_S = ("mobilevit_s", 640, Implementation.VIT)

    def build(self, dims: int):
        impl = self.value[2]
        if impl == Implementation.CLASSY_VISION:
            model = build_model({"name": self.value[0]})
            # Remove head exec wrapper, which we don't need, and breaks pickling
            # (needed for spawn dataloaders).
            return model.classy_model
        if impl == Implementation.TORCHVISION:
            try:
                fn = self.value[0]
                if self.name.startswith("TV_RESNET"):
                    return fn(num_classes=dims, zero_init_residual=True)
                else:
                    return fn(num_classes=dims)
            except Exception as e:
                raise AssertionError("Model implementation not handled: %s (%s)" % (self.name, str(e)))


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class Model(LightningModule):
    def __init__(self, backbone: str, dims: int, pool_param: float):
        super().__init__()
        self.backbone_type = Backbone[backbone]
        self.backbone = self.backbone_type.build(dims=dims)
        impl = self.backbone_type.value[2]
        if impl == Implementation.CLASSY_VISION:
            self.embeddings = nn.Sequential(
                GlobalGeMPool2d(pool_param),
                nn.Linear(self.backbone_type.value[1], dims),
                L2Norm(),
            )
        elif backbone == "TV_RESNET50":
            self.backbone.avgpool = GlobalGeMPool2d(pool_param)
            
        elif backbone == 'TV_EFFICIENTNET_B0':
            self.backbone.avgpool = GlobalGeMPool2d(pool_param)
            self.backbone.fc = nn.Identity()
            self.embeddings = L2Norm()
        elif backbone == 'TV_MOBILENETV2':
            self.backbone = MobilenetV2()
            self.embeddings = L2Norm()
        elif backbone == 'TV_MOBILENETV3':
            self.backbone.avgpool = GlobalGeMPool2d(pool_param)
            self.embeddings = L2Norm()

        elif backbone == 'TV_REGNET_Y_800MF':
            self.backbone.avgpool = GlobalGeMPool2d(pool_param)
            self.backbone.fc = nn.Identity()
            self.embeddings = nn.Sequential(
                nn.Linear(784, 512),
                L2Norm()
            )

        elif impl == Implementation.TIMM:
            if self.backbone_type.value[1] != dims:
                self.backbone.global_pool = GlobalGeMPool2d(pool_param)
                self.backbone.fc = nn.Identity()
                self.backbone.classifier = nn.Identity()
                self.embeddings = nn.Sequential(
                    nn.Linear(self.backbone_type.value[1], dims),
                    L2Norm(),
                )
            else: # embedding size is 512
                self.backbone.global_pool = GlobalGeMPool2d(pool_param)
                # self.backbone.fc = nn.Identity()
                self.backbone.fc = nn.Linear(512, 512)
                self.backbone.classifier = nn.Identity()
                self.embeddings = L2Norm()

        elif impl == Implementation.VIT:
            self.backbone.head.global_pool = GlobalGeMPool2d(pool_param, keep_res=False)
            self.backbone.head.fc = nn.Linear(self.backbone_type.value[1], dims)
            self.embeddings = L2Norm()


    def forward(self, x, mode=None):
        if isinstance(self.backbone, MobilenetV2):
            return self.backbone(x, mode="embedding_only")
        x = self.backbone(x)
        if mode == 'teacher':
            return x
        return self.embeddings(x)
    # def forward(self, x, mode=None):
    #     x = self.backbone(x)
    #     if mode == 'teacher':
    #         return x
    #     else:
    #         return self.embeddings(x)


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group("Model")
        parser.add_argument(
            "--backbone", default="TV_RESNET18", choices=[b.name for b in Backbone]
        )
        parser.add_argument("--dims", default=512, type=int)
        parser.add_argument("--pool_param", default=3, type=float)