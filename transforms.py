import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import create_transform

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

def get_transforms(img_size):  # 'original', 'rand-m9-mstd0.5-inc1', 등
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