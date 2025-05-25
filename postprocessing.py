def load_confusion_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cls1, cls2 = line.strip().split()
            pairs.append((cls1, cls2))
    return pairs

def redistribute_probs_csv(confusion_pairs_file, pred_probs_file, output_file):
    # 혼동 쌍 로드
    confusion_pairs = load_confusion_pairs(confusion_pairs_file)
    
    # 예측 확률 csv 로드
    df = pd.read_csv(pred_probs_file)
    
    # 클래스명 리스트 (csv 첫 행 헤더)
    class_names = df.columns.tolist()
    
    # 확률 행렬 (numpy)
    probs = df.to_numpy()
    
    # 클래스명 → 인덱스 매핑
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # 확률 재분배
    for cls1, cls2 in confusion_pairs:
        if cls1 in class_to_idx and cls2 in class_to_idx:
            idx1, idx2 = class_to_idx[cls1], class_to_idx[cls2]
            combined = probs[:, idx1] + probs[:, idx2]
            avg = combined / 2
            probs[:, idx1] = avg
            probs[:, idx2] = avg
        else:
            print(f"경고: {cls1} 또는 {cls2}가 클래스 목록에 없습니다.")
    
    # 재분배된 확률로 DataFrame 생성 (원래 컬럼명 유지)
    df_adjusted = pd.DataFrame(probs, columns=class_names)
    
    # 결과 csv 저장
    df_adjusted.to_csv(output_file, index=False)
    print(f"조정된 확률이 {output_file}에 저장되었습니다.")

# 사용 예시
redistribute_probs_csv("confusion_pairs_20.txt", "./submit/densenet169_train5_submission.csv", "./submit/densenet169_train5_softLabel_20_submission.csv")