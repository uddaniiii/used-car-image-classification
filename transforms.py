import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import create_transform
from timm.data.mixup import Mixup

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

# timm 사용

def get_transforms(img_size):  # 'original', 'rand-m9-mstd0.5-inc1', 'augmix-m5-w4-d2', 'trivial', 'rand-m9-n3-mstd0.5'
    train_transform = create_transform(
        input_size=img_size,
        is_training=True,
        # auto_augment='augmix-m5-w4-d2',
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

# 1. CutMix 단독
cutmix_only = Mixup(
    mixup_alpha=0.0,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.0,
    mode='batch',
    label_smoothing=0.0,
    num_classes=396
)

# 2. MixUp 단독
mixup_only = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=0.0,
    prob=1.0,
    switch_prob=0.0,
    mode='batch',
    label_smoothing=0.0,
    num_classes=396
)

# 3. CutMix + MixUp 혼합
mixed = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.0,
    num_classes=396
)