# 🚗 Used Car Classification

중고차 이미지 기반의 차종 분류 모델입니다.  
396개 차량 클래스를 분류하며, 실전 성능과 도메인 적합성을 고려한 다양한 실험을 진행했습니다.

## 🔍 프로젝트 개요
- **문제 정의**: 중고차 이미지에서 차량 차종 분류 (396클래스)
- **사용 데이터**: 총 33,137장 학습 / 8,258장 테스트

## 🏆 성능
- **Best Leaderboard Score**: 0.1541 (앙상블)

## 📁 프로젝트 폴더 구조
```
📦used-car-image-classification
 ┣ 📂notebooks                    # 데이터 탐색 및 시각화 노트북
 ┃ ┣ 📜EDA.ipynb                 # 데이터 통계 및 분포 분석
 ┃ ┣ 📜analysis.ipynb            # 모델 성능 및 오류 분석
 ┃ ┗ 📜visualize_preprocessing.ipynb # 전처리 시각화
 ┣ 📂scripts                     # 데이터 전처리 / 후처리 스크립트
 ┃ ┣ 📜outlier_detection.py      # 이상치 탐지 및 제거
 ┃ ┣ 📜postprocess_soft_label.py # soft label 후처리
 ┃ ┣ 📜remove_duplicates_from_train.py # 중복 이미지 제거
 ┃ ┗ 📜save_class_names.py       # 클래스 이름 저장
 ┣ 📜config.py                   # 실험 설정 파일
 ┣ 📜dataset.py                  # 커스텀 Dataset 정의
 ┣ 📜ensemble.py                 # 앙상블 예측 스크립트
 ┣ 📜hem.py                      # Hard Example Mining 관련 모듈
 ┣ 📜inference.py                # 모델 추론 스크립트
 ┣ 📜losses.py                   # Loss 함수 정의
 ┣ 📜main.py                     # 전체 학습/추론 통합 실행 파일
 ┣ 📜model.py                    # 모델 아키텍처 정의
 ┣ 📜optuna_study.py             # Optuna 기반 하이퍼파라미터 튜닝
 ┣ 📜requirements.txt            # 필요 라이브러리 목록
 ┣ 📜train.py                    # 모델 학습 실행 스크립트
 ┣ 📜transforms.py               # 데이터 증강 및 전처리 정의
 ┗ 📜utils.py                    # 공통 유틸 함수 모음
```
