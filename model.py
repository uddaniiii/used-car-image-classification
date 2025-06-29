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
        
        # efficientNet
        self.feature_dim = self.backbone.classifier[1].in_features  # EfficientNet 구조 기준
        self.backbone.classifier = nn.Identity()  # feature extractor로 사용

        # # densenet        
        # self.feature_dim = self.backbone.classifier.in_features  

        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    def forward(self, x): # -> resnet, swin, efficient
        x = self.backbone(x)       
        x = self.head(x) 
        return x

    # def forward(self, x): # -> convnext
    #     x = self.backbone(x)       # [B, C, 1, 1]
    #     x = x.view(x.size(0), -1)  # or x = torch.flatten(x, 1)
    #     x = self.head(x)           # [B, num_classes]
    #     return x

    # def forward(self, x): # -> densenet
    #     features = self.backbone.features(x)           # [B, C, H, W]
    #     out = F.relu(features, inplace=True)
    #     out = F.adaptive_avg_pool2d(out, (1, 1))       # [B, C, 1, 1]
    #     out = torch.flatten(out, 1)                    # [B, C]
    #     out = self.head(out)                           # [B, num_classes]
    #     return out

class CosineSimilarityClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)               # feature 정규화
        w = F.normalize(self.weight, p=2, dim=1)     # weight 정규화
        logits = self.scale * torch.matmul(x, w.T)   # cosine similarity × scaling
        return logits

class BAPClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, attention_heads=32):
        super().__init__()
        self.attention_heads = attention_heads
        self.attention_conv = nn.Conv2d(in_channels, attention_heads, kernel_size=1)
        self.classifier = nn.Linear(in_channels * attention_heads, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        A = self.attention_conv(x)  # [B, attention_heads, H, W]
        A = torch.sigmoid(A)        # attention weights [0,1]

        # element-wise multiply and pool spatially
        bap_features = []
        for i in range(self.attention_heads):
            a = A[:, i, :, :].unsqueeze(1)  # [B,1,H,W]
            weighted_feature = x * a         # [B,C,H,W]
            pooled = F.adaptive_avg_pool2d(weighted_feature, (1,1)).squeeze(-1).squeeze(-1)  # [B,C]
            bap_features.append(pooled)
        bap_features = torch.cat(bap_features, dim=1)  # [B, C*attention_heads]

        out = self.classifier(bap_features)  # [B, num_classes]
        return out
    
class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, scale=30.0, attention_heads=32):
        super(TimmModel, self).__init__()
        # base
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  
        self.feature_dim = self.backbone.num_features
        # self.head = nn.Linear(self.feature_dim, num_classes) # base
        self.head = CosineSimilarityClassifier(self.feature_dim, num_classes, scale=scale) # coine similarity classifier

        # bap
        # self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        # self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # 마지막 피처맵 채널 수
        # self.bap = BAPClassifier(self.feature_dim, num_classes, attention_heads)

        # global local concat
        # self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')  # output [B, C, H, W]
        # self.feature_dim = self.backbone.num_features

        # # Global feature: GAP
        # self.global_pool = nn.AdaptiveAvgPool2d(1)

        # # Local feature: Conv + GAP
        # self.local_conv = nn.Sequential(
        #     nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=1),
        #     nn.BatchNorm2d(self.feature_dim // 2),
        #     nn.ReLU(inplace=True),
        # )
        # self.local_pool = nn.AdaptiveAvgPool2d(1)

        # self.classifier = CosineSimilarityClassifier(
        #     in_features=self.feature_dim + self.feature_dim // 2,
        #     num_classes=num_classes,
        #     scale=scale,
        # )

    def forward(self, x):
        # base
        x = self.backbone(x)
        x = self.head(x)
        return x

        # bap
        # features = self.backbone(x)  # list of feature maps
        # x = features[-1]  # 가장 마지막 feature map: [B, C, H, W]
        # x = self.bap(x)   # 이제 conv2d 입력에 맞음
        # return x

        # global local concat
        # feat = self.backbone(x)  # [B, C, H, W]

        # global_feat = self.global_pool(feat).squeeze(-1).squeeze(-1)  # [B, C]
        # local_feat = self.local_conv(feat)  # [B, C//2, H, W]
        # local_feat = self.local_pool(local_feat).squeeze(-1).squeeze(-1)  # [B, C//2]

        # combined_feat = torch.cat([global_feat, local_feat], dim=1)  # [B, C + C//2]
        # out = self.classifier(combined_feat)

        # return out
