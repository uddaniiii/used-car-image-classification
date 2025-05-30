import shutil
import os
from collections import defaultdict
import hashlib

# 1. 원본 train 폴더 → 새 폴더로 복사
src_train_dir = 'data/train'
backup_train_dir = 'data/train_no_duplicates'

if not os.path.exists(backup_train_dir):
    shutil.copytree(src_train_dir, backup_train_dir)
    print(f"✅ '{src_train_dir}' 폴더를 '{backup_train_dir}'로 복사 완료")
else:
    print(f"⚠️ '{backup_train_dir}' 폴더가 이미 존재합니다. 중복 제거 작업 진행합니다.")

# 2. 새 폴더 내 모든 이미지 해시 계산 및 중복 탐지
hash_to_paths = defaultdict(list)

for root, dirs, files in os.walk(backup_train_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            hash_to_paths[img_hash].append(file_path)

# 3. 중복 이미지 제거 (첫번째만 남기고 나머지 삭제)
dup_count = 0
for paths in hash_to_paths.values():
    if len(paths) > 1:
        # 첫 번째 이미지 제외하고 삭제
        for dup_path in paths[1:]:
            os.remove(dup_path)
            dup_count += 1

print(f"🗑️ 중복 이미지 총 {dup_count}개 제거 완료")
