import os


def save_classes_to_file(root_dir, output_file='classes.txt'):
    classes = sorted(os.listdir(root_dir))
    with open(output_file, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"✅ 클래스명 {len(classes)}개를 {output_file}에 저장 완료")

if __name__ == "__main__":
    data_root = './'  # 네 데이터셋 학습 폴더 경로로 바꿔줘
    save_classes_to_file(data_root)