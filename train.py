import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import log_loss
import torch.nn.functional as F
from transforms import cutmix_only, mixup_only, mixed
import numpy as np 

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # # CutMix / MixUp 적용
        # images, labels = cutmix_only(images, labels)

        optimizer.zero_grad()
        outputs = model(images) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)

        # # CutMix / MixUp 적용
        # labels_idx = torch.argmax(labels, dim=1)  # soft label -> 정수 인덱스

        correct += (preds == labels).sum().item() # base
        # correct += (preds == labels_idx).sum().item() # cutmix/mixup

        total += labels.size(0)

    avg_loss = train_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, criterion, device, class_names):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) 
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(dataloader)
    accuracy = 100 * correct / total

    logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
    return avg_loss, accuracy, logloss
