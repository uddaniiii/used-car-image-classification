import pandas as pd
import os

# CSV 파일 읽기
df = pd.read_csv("./outlier_verification_gui.csv")

# image_path의 "train"을 "train_outlier"로 바꿈
df["image_path"] = df["image_path"].str.replace("train", "train_outlier", regex=False)

# label이 'y'인 행 필터링
to_delete = df[df["label"] == 'y']

# 해당 이미지 파일 삭제
for path in to_delete["image_path"]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")
    else:
        print(f"File not found: {path}")