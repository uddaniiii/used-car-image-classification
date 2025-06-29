import torch
from torch.utils.data import DataLoader, Subset
import os
import config
from dataset import CustomImageDataset
from model import TimmModel
from train import train_one_epoch, validate_one_epoch
from utils import seed_everything
from losses import get_loss_fn
from tqdm import tqdm
from transforms import get_transforms
from sklearn.model_selection import train_test_split
from utils import seed_everything, load_class_names
from sklearn.metrics import log_loss
import torch.nn.functional as F

def calculate_sample_losses(model, dataset, device, criterion):
    model.eval()
    losses = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=config.NUM_WORKERS)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calculating sample losses"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_losses = criterion(outputs, labels)  # sample-wise loss
            losses.extend(batch_losses.cpu().tolist())
    return losses

def select_hard_examples(losses, threshold):
    hard_indices = [i for i, loss in enumerate(losses) if loss > threshold]
    print(f"Selected {len(hard_indices)} hard examples with loss > {threshold}")
    return hard_indices

def main():
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RandAug 포함 transform (train용), 미포함 transform (val용) 분리
    train_transform, val_transform = get_transforms(config.IMG_SIZE)

    # 1. 평가용 dataset (RandAug 없이)
    full_dataset_for_eval = CustomImageDataset(config.TRAIN_DIR, transform=val_transform)
    print(f"Total train images: {len(full_dataset_for_eval)}")

    targets = [label for _, label in full_dataset_for_eval.samples]

    # Stratified Split
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=42
    )

    val_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=val_transform), val_idx)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # 2. 모델 정의 및 best 모델 weight 로드
    model = TimmModel(model_name=config.MODEL_NAME, num_classes=len(full_dataset_for_eval.class_to_idx))
    model.load_state_dict(torch.load("./checkpoints/tiny_vit_21m_384_randaug_cosine/62_0.2205_0.0780_0.0783.pth"))
    model.to(device)

    class_names = load_class_names('./classes.txt')

    # 3. Loss 함수 정의 (reduction='none' 필수)
    criterion = get_loss_fn(config.LOSS, reduction='none')

    # 4. 샘플별 loss 계산
    sample_losses = calculate_sample_losses(model, full_dataset_for_eval, device, criterion)

    # 5. threshold 기준 hard example 선택
    hard_indices = select_hard_examples(sample_losses, threshold=0.5)
    if len(hard_indices) == 0:
        print("No hard examples found. Exiting.")
        return

    # 6. RandAug 포함된 dataset으로 다시 로드해서 subset 생성
    full_dataset_for_train = CustomImageDataset(config.TRAIN_DIR, transform=train_transform)
    hard_dataset = Subset(full_dataset_for_train, hard_indices)
    hard_loader = DataLoader(hard_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    # 7. Optimizer 정의 (lr 줄여서 fine-tune)
    finetune_lr = config.LR * 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    # 8. Fine-tuning loop with Early Stopping
    finetune_epochs = 50
    best_val_logloss = float('inf')
    epochs_no_improve = 0
    patience = 5  # 5 epoch 개선 없으면 중단

    for epoch in range(finetune_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(hard_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss = loss.mean()  # reduction='none'이므로 평균으로 변환
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / len(hard_loader)
        accuracy = 100 * correct / total
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images) 
                loss = criterion(outputs, labels)
                val_loss += loss.mean().item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
        
        print(f"[{epoch+1}] "
            f"Train Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} || "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | LogLoss: {val_logloss:.4f}")

        # Early Stopping 체크
        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            epochs_no_improve = 0
            # 여기서 필요하면 best 모델 저장도 가능
            filename = f"{epoch+1}_{train_loss:.4f}_{val_loss:.4f}_{val_logloss:.4f}.pth"
            os.makedirs(config.SAVE_DIR, exist_ok=True)  # 디렉토리 없으면 생성
            save_path = os.path.join(config.SAVE_DIR, filename)
            torch.save(model.state_dict(), save_path)
            print(f"📦 Best model saved at epoch {epoch+1} ({save_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Validation log loss hasn't improved for {patience} epochs. Early stopping...")
                break

if __name__ == "__main__":
    main()
