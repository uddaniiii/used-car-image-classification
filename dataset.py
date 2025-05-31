import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image 

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            # 테스트셋: 라벨 없이 이미지 경로만 저장
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            # 학습셋: 클래스별 폴더 구조에서 라벨 추출
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    # # albumentation 용
    # def __getitem__(self, idx):
    #     if self.is_test:
    #         img_path = self.samples[idx][0]
    #         # image = Image.open(img_path).convert('RGB')
    #         image = imread_unicode(img_path)  # 수정됨
    #         if self.transform:
    #             # image = self.transform(image)
    #             image = self.transform(image=image)['image']
    #         return image
    #     else:
    #         img_path, label = self.samples[idx]
    #         # image = Image.open(img_path).convert('RGB')
    #         image = imread_unicode(img_path)  # 수정됨
    #         if self.transform:
    #             # image = self.transform(image)
    #             image = self.transform(image=image)['image']
    #         return image, label

    # timm 용
    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = imread_unicode(img_path)  # numpy.ndarray
            image = Image.fromarray(image)    # numpy → PIL
            if self.transform:
                image = self.transform(image)  # timm transform은 PIL 이미지 사용
            return image
        else:
            img_path, label = self.samples[idx]
            image = imread_unicode(img_path)  # numpy.ndarray
            image = Image.fromarray(image)    # numpy → PIL
            if self.transform:
                image = self.transform(image)
            return image, label
        
    # # image, label, img_path도 반환해야할때
    # def __getitem__(self, idx):
    #     if self.is_test:
    #         img_path = self.samples[idx][0]
    #         image = imread_unicode(img_path)
    #         if self.transform:
    #             image = self.transform(image=image)['image']
    #         return image
    #     else:
    #         img_path, label = self.samples[idx]
    #         image = imread_unicode(img_path)
    #         if self.transform:
    #             image = self.transform(image=image)['image']
    #         return image, label, img_path  # ✅ 경로도 함께 반환!