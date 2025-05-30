import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd 
import umap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.models as models
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from dataset import CustomImageDataset
from config import TRAIN_DIR, IMG_SIZE, BATCH_SIZE, MODEL_NAME, SEED
from transforms import get_transforms
from model import BaseModel, TimmModel

def extract_features(model, dataloader, device):
    model.eval()
    all_features, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu())
            all_labels.extend(labels)
            all_paths.extend(paths)

    features = torch.cat(all_features).numpy()
    return features, np.array(all_labels), np.array(all_paths)

def get_outliers(features, labels, paths, top_n=5):
    outlier_info = []

    for cls in np.unique(labels):
        cls_idxs = np.where(labels == cls)[0]
        cls_feats = features[cls_idxs]

        dist_matrix = cosine_distances(cls_feats)
        mean_dists = dist_matrix.mean(axis=1)

        top_outlier_idxs = mean_dists.argsort()[::-1][:top_n]
        for idx in top_outlier_idxs:
            outlier_info.append({
                "class": cls,
                "image_path": paths[cls_idxs[idx]],
                "mean_cosine_distance": mean_dists[idx]
            })


    return pd.DataFrame(outlier_info)

def get_outliers_global(features, paths, top_n=20):
    # 모든 샘플 간 코사인 거리 행렬 계산
    dist_matrix = cosine_distances(features)
    # 각 샘플별로 다른 샘플과의 평균 거리 계산 (자기 자신 제외)
    mean_dists = (dist_matrix.sum(axis=1) - 0) / (dist_matrix.shape[1] - 1)  # 자기 자신 제외
    
    top_outlier_idxs = mean_dists.argsort()[::-1][:top_n]
    outlier_info = []
    for idx in top_outlier_idxs:
        outlier_info.append({
            "image_path": paths[idx],
            "mean_cosine_distance": mean_dists[idx]
        })
    return pd.DataFrame(outlier_info)

def visualize_umap(features, labels, seed=42, title="UMAP projection"):
    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(scatter, label="Class Label")
    plt.title(title)
    plt.show()

def visualize_outliers(outlier_df, top_n=20, cols=3, img_size=(6,6)):
    import math
    top_outliers = outlier_df.sort_values(by='mean_cosine_distance', ascending=False).head(top_n)
    rows = math.ceil(top_n / cols)

    for row_idx in range(rows):
        plt.figure(figsize=(img_size[0]*cols, img_size[1]))
        for col_idx in range(cols):
            idx = row_idx * cols + col_idx
            if idx >= len(top_outliers):
                break
            plt.subplot(1, cols, col_idx + 1)
            img = mpimg.imread(top_outliers.iloc[idx]['image_path'])
            plt.imshow(img)
            plt.axis('off')
            cls = top_outliers.iloc[idx]['class']
            dist = top_outliers.iloc[idx]['mean_cosine_distance']
            plt.title(f"Class: {cls}\nDist: {dist:.3f}", fontsize=12)
        plt.tight_layout()
        plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
num_classes = 396  
model = BaseModel(model_name=MODEL_NAME, num_classes=num_classes, pretrained=False)
model.load_state_dict(torch.load('./checkpoints/densenet169_base/22_0.0232_0.1602.pth', map_location=device))
model.to(device)
model.eval()

_, val_transform = get_transforms(IMG_SIZE)
# 전체 학습 데이터셋 로딩
full_dataset_for_feat = CustomImageDataset(TRAIN_DIR, transform=val_transform)
full_loader = DataLoader(full_dataset_for_feat, batch_size=BATCH_SIZE, shuffle=False)

# Feature 추출
features, labels, paths = extract_features(model, full_loader, device)

# 시각화
visualize_umap(features, labels, seed=SEED, title="UMAP projection of Train Set")

# 이상치 추출
outlier_df = get_outliers(features, labels, paths, top_n=5)
top_outliers = outlier_df.sort_values(by='mean_cosine_distance', ascending=False)
print(outlier_df.head())

outlier_df_global = get_outliers_global(features, paths, top_n=20)
print(outlier_df_global.head())

# 저장
top_outliers.to_csv("outliers.csv", index=False)
outlier_df_global.to_csv("global_outliers.csv", index=False)
