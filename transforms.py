import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        # A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform