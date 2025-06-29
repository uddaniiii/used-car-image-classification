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
from losses import get_loss_fn
import optuna


def parse_args():
    parser = argparse.ArgumentParser(description="used-car-image-classification")

    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--loss", type=str, default=config.LOSS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")

    return parser.parse_args()


def objective(trial=None):
    # optuna trial이 있으면 그걸로 하이퍼파라미터 선택, 없으면 argparse로 받음
    if trial is not None:
        # base
        img_size = config.IMG_SIZE
        batch_size = config.BATCH_SIZE
        lr = config.LR
        loss_name = config.LOSS
        epochs = config.EPOCHS
        seed = config.SEED

        # optuna
        # img_size = trial.suggest_categorical("img_size", [224, 256, 288])
        # batch_size = trial.suggest_categorical("batch_size", [64, 32, 16, 8])
        scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "reduce_on_plateau"])
    else:
        args = parse_args()
        lr = args.lr
        batch_size = args.batch_size
        img_size = args.img_size
        loss_name = config.LOSS
        epochs = config.EPOCHS
        seed = config.SEED

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="used-car-classification",
        name=f"trial_{trial.number}" if trial else config.MODEL_PATH,
        config={
            "IMG_SIZE": img_size,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "LEARNING_RATE": lr,
            "SEED": seed,
            "LOSS": loss_name,
        },
        reinit=True,
    )

    try:
        full_dataset = CustomImageDataset(config.TRAIN_DIR, transform=None)
        targets = [label for _, label in full_dataset.samples]

        train_idx, val_idx = train_test_split(
            range(len(targets)), test_size=0.2, stratify=targets, random_state=42
        )

        train_transform, val_transform = get_transforms(img_size)

        train_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=train_transform), train_idx)
        val_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=val_transform), val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

        model = TimmModel(model_name=config.MODEL_NAME, num_classes=len(train_dataset.dataset.class_to_idx)).to(device)
        class_names = list(train_dataset.dataset.class_to_idx.keys())

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 스케줄러 선택
        if scheduler_name == "cosine":
            T_max = trial.suggest_int("T_max", 5, epochs)
            eta_min = trial.suggest_float("eta_min", 1e-6, 1e-3, log=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )

        elif scheduler_name == "reduce_on_plateau":
            factor = trial.suggest_float("plateau_factor", 0.3, 0.7)
            patience = trial.suggest_int("plateau_patience", 3, 7)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience
            )

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")        
        
        criterion = get_loss_fn(loss_name)
        best_logloss = float("inf")
        patience = 10
        counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_logloss = validate_one_epoch(model, val_loader, criterion, device, class_names)

            # 스케줄러 업데이트
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            wandb.log(
                {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Valid Loss": val_loss,
                    "Valid Accuracy": val_acc,
                    "Validation LogLoss": val_logloss,
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

            print(
                f"[{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | LogLoss: {val_logloss:.4f}"
            )

            if val_logloss < best_logloss:
                best_logloss = val_logloss
                trial_num_str = f"trial{trial.number}_" if trial else ""
                filename = f"{trial_num_str}{epoch+1}_{train_loss:.4f}_{val_loss:.4f}_{val_logloss:.4f}.pth"
                os.makedirs(config.SAVE_DIR, exist_ok=True)
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

            # Optuna pruner에게 중간 결과 report
            if trial:
                trial.report(val_logloss, epoch)
                if trial.should_prune():
                    print("⚠️ Trial pruned by Optuna.")
                    raise optuna.exceptions.TrialPruned()

    finally:
        wandb.finish()

    return best_logloss


def main():
    args = parse_args()

    # Optuna 스터디 생성 및 실행
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, timeout=None)

    print("==== Optuna study finished ====")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (logloss): {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
