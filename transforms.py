import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import create_transform
from timm.data.mixup import Mixup
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

# Albumentations 사용

# def get_transforms(img_size):
#     train_transform = A.Compose([
#         A.Resize(img_size, img_size),
#         # A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8),
#         # A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])
#     val_transform = A.Compose([
#         A.Resize(img_size, img_size),
#         A.Normalize(mean=(0.485, 0.456, 0.406), 
#                     std=(0.229, 0.224, 0.225)),
#         ToTensorV2()
#     ])

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img = np.array(img)  # PIL -> ndarray

        if img.ndim == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        else:
            # grayscale일 경우 (드물지만 대비)
            img_clahe = self.clahe.apply(img)

        return Image.fromarray(img_clahe)

# timm 사용

def get_transforms(img_size):  # 'original', 'rand-m9-mstd0.5-inc1', 'augmix-m5-w4-d2', 'trivial', 'rand-m9-n3-mstd0.5'
    # base

    train_transform = create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',  # Transformer 계열은 bicubic이 안정적
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    val_transform = create_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    return train_transform, val_transform

    # CLAHE
    
    # clahe = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))

    # train_transform = transforms.Compose([
    #     clahe,
    #     create_transform(
    #         input_size=img_size,
    #         is_training=True,
    #         # auto_augment='rand-m9-mstd0.5-inc1',  # CLAHE 효과만 보려면 꺼둬야 정확
    #         interpolation='bicubic',
    #         mean=(0.485, 0.456, 0.406),
    #         std=(0.229, 0.224, 0.225)
    #     )
    # ])

    # val_transform = transforms.Compose([
    #     clahe,
    #     create_transform(
    #         input_size=img_size,
    #         is_training=False,
    #         interpolation='bicubic',
    #         mean=(0.485, 0.456, 0.406),
    #         std=(0.229, 0.224, 0.225)
    #     )
    # ])

    # return train_transform, val_transform

# 1. CutMix 단독
cutmix_only = Mixup(
    mixup_alpha=0.0,
    cutmix_alpha=0.4,       # 더 작고 안정적인 영역 크기
    prob=0.5,               # 절반만 CutMix
    switch_prob=0.0,
    mode='elem',            # 다양한 패턴 유도
    label_smoothing=0.0,
    num_classes=396
)

# 2. MixUp 단독
mixup_only = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=0.0,
    prob=0.7,                # 70% 확률로 MixUp 적용
    switch_prob=0.0,
    mode='elem',             # 각 샘플별 MixUp 적용
    label_smoothing=0.0,
    num_classes=396
)

# 3. CutMix + MixUp 혼합
mixed = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=0.4,
    prob=0.6,
    switch_prob=0.5,
    mode='elem',
    label_smoothing=0.0,
    num_classes=396
)