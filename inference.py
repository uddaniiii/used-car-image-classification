import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomImageDataset
import config
from transforms import get_transforms
from model import BaseModel, TimmModel
from utils import seed_everything, load_class_names
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="used-car-image-classification")
    parser.add_argument("--weight_path", type=str, required=True, help="학습된 모델 가중치 경로")
    parser.add_argument("--output_csv", type=str, default=config.OUTPUT_CSV, help="출력 제출 파일명")
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=64, help="배치 사이즈")
    parser.add_argument("--seed", type=int, default=42, help="시드")
    return parser.parse_args()

def main():
    args = parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names('./classes.txt')

    train_transform, val_transform = get_transforms(args.img_size)
    test_dataset = CustomImageDataset(config.TEST_DIR, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TimmModel(model_name=config.MODEL_NAME, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference Progress"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            # 각 배치의 확률을 리스트로 변환
            for prob in probs.cpu():  # prob: (num_classes,)
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)

    pred = pd.DataFrame(results)

    submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')
    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(args.output_csv, index=False, encoding='utf-8-sig')

    print(f"✅ Inference 완료. 확률 기반 결과 저장됨: {args.output_csv}")

if __name__ == "__main__":
    main()
