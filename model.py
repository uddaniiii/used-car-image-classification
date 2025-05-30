import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModelForImageClassification
import os

class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(BaseModel, self).__init__()
        self.backbone = getattr(models, model_name)(weights='DEFAULT' if pretrained else None)  # torchvision 모델 불러오기
        # # resnet
        # self.feature_dim = self.backbone.fc.in_features 
        # self.backbone.fc = nn.Identity()  # feature extractor로만 사용
        
        # convnext
        # self.feature_dim = self.backbone.classifier[2].in_features  
        # self.backbone.classifier = nn.Identity()  # classifier 대신 feature extractor로만 쓰기 위해 제거
        
        # swinTransformer tiny
        # self.feature_dim = self.backbone.head.in_features  # swin은 head가 마지막 fc
        # self.backbone.head = nn.Identity()  # feature extractor로만 사용
        
        # # efficientNet
        # self.feature_dim = self.backbone.classifier[1].in_features  # EfficientNet 구조 기준
        # self.backbone.classifier = nn.Identity()  # feature extractor로 사용

        # densenet        
        self.feature_dim = self.backbone.classifier.in_features  

        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    # def forward(self, x): # -> resnet, swin, efficient
    #     x = self.backbone(x)       
    #     x = self.head(x) 
    #     return x

    # def forward(self, x): # -> convnext
    #     x = self.backbone(x)       # [B, C, 1, 1]
    #     x = x.view(x.size(0), -1)  # or x = torch.flatten(x, 1)
    #     x = self.head(x)           # [B, num_classes]
    #     return x

    def forward(self, x): # -> densenet
        features = self.backbone.features(x)           # [B, C, H, W]
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))       # [B, C, 1, 1]
        out = torch.flatten(out, 1)                    # [B, C]
        out = self.head(out)                           # [B, num_classes]
        return out
    
class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(TimmModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  
        # base
        self.feature_dim = self.backbone.num_features
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        x = self.head(x)
        return x