import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from dataset import CustomImageDataset
from train import train_one_epoch, validate_one_epoch
import config
from utils import seed_everything
from transforms import get_transforms
from model import BaseModel, TimmModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os

def parse_args():
    parser = argparse.ArgumentParser(description="used-car-image-classification")

    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--seed", type=int, default=config.SEED)

    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="used-car-classification",
        name=config.MODEL_PATH,
        config={
            'IMG_SIZE': args.img_size,
            'BATCH_SIZE': args.batch_size,
            'EPOCHS': args.epochs,
            'LEARNING_RATE': args.lr,
            'SEED' : args.seed
        }
    )

    # 전체 데이터셋 로드
    full_dataset = CustomImageDataset(config.TRAIN_DIR, transform=None)
    print(f"총 이미지 수: {len(full_dataset)}")

    targets = [label for _, label in full_dataset.samples]

    # Stratified Split
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=42
    )

    train_transform, val_transform = get_transforms(args.img_size)

    # Subset + transform 각각 적용
    train_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=train_transform), train_idx)
    val_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=val_transform), val_idx)
    print(f'train 이미지 수: {len(train_dataset)}, valid 이미지 수: {len(val_dataset)}')

    # DataLoader 정의
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    # 실험할 때 확인할 것
    model = TimmModel(model_name=config.MODEL_NAME, num_classes=len(train_dataset.dataset.class_to_idx)).to(device)
    # print(model)
    class_names = list(train_dataset.dataset.class_to_idx.keys())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_logloss = float('inf')
    patience = 8
    counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_logloss = validate_one_epoch(model, val_loader, criterion, device, class_names)

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Valid Loss": val_loss,
            "Valid Accuracy": val_acc,
            "Validation LogLoss": val_logloss,
            "LR": optimizer.param_groups[0]['lr'],
        })

        print(f"[{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | LogLoss: {val_logloss:.4f}")

        # Best model 저장 + 조기 종료 체크
        if val_logloss < best_logloss:
            best_logloss = val_logloss

            filename = f"{epoch+1}_{train_loss:.4f}_{val_loss:.4f}_{val_logloss:.4f}.pth"
            os.makedirs(config.SAVE_DIR, exist_ok=True)  # 디렉토리 없으면 생성
            save_path = os.path.join(config.SAVE_DIR, filename)
            torch.save(model.state_dict(), save_path)
            print(f"📦 Best model saved at epoch {epoch+1} ({save_path})")

            counter = 0
        else:
            counter += 1
            print(f"🕒 No improvement in logloss for {counter} epoch(s).")

            if counter >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1}. Best logloss: {best_logloss:.4f}")
                break

if __name__ == "__main__":
    main()
