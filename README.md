# Tennessee Eastman Process (TEP) 공정 이상 탐지 및 정상화 파이프라인

## 팀 정보
**2025 AI 커리어패스 프로그램 - 팀3**
- 국민대학교 박규민
- 동양미래대학교 방석영
- 세종대학교 엄태호
- 한양여자대학교 조유영

## 프로젝트 개요

**4단계 공정 이상 분석 및 정상화 시스템**
- Model1: Fault 시점 탐지 + Fault 종류 분류 (CNN1D2D Discriminator)
- Model2: 조작 변수 정상화 (Conditional TCN-AE) - **추후 추가 예정**
- Model3: 반응 변수 예측 (RSSM) - **추후 추가 예정**
- Model4: 정상 여부 재분류 (Model1 재사용)

**주요 특징**
- 52개 센서 데이터 기반 시계열 분석
- 21가지 결함 유형 분류 (정상상태 포함)
- 슬라이딩 윈도우 기반 배치 처리 (50, 10)
- 반복 정상화 파이프라인 (최대 3회)
- LLM 기반 결과 해설 (추후 활성화 예정)

## 프로젝트 구조

```
tennessee_eastman_diploma/
├── src/
│   ├── main/              # 메인 파이프라인
│   │   ├── pipeline.py    # TEP 전체 파이프라인
│   │   └── model1_module.py  # Model1 모듈
│   ├── model1/            # Model1 관련 코드
│   │   ├── convolutional_models.py    # CNN1D2D 모델
│   │   ├── evaluate_model.py          # 모델 평가
│   │   └── train_model.py             # 모델 훈련
│   ├── data/              # 데이터 처리 및 로딩
│   │   └── dataset.py     # TEP 데이터셋 클래스들
│   └── LLM/               # LLM 관련 코드 (추후 활성화)
├── data/                  # 데이터셋 저장소
│   ├── final_X.npy        # 전처리된 입력 데이터
│   └── final_Y.npy        # 전처리된 라벨 데이터
├── model_pretrained/      # 사전 훈련된 모델
│   └── model1/
│       └── 30_epoch_checkpoint.pth
├── logs/                  # 로그 파일
└── setup.py               # 프로젝트 설정 파일
```

## 데이터셋 정보

**Tennessee Eastman Process 데이터**
- 센서 개수: 52개 (22개 공정 측정값, 19개 분석 측정값, 11개 조작 변수)
- 결함 유형: 21가지 (정상상태 포함)
- 샘플링 주기: 3분

**슬라이딩 윈도우 처리**
- 원본 데이터: 0~959 시점 (960개)
- 윈도우 크기: 50
- 스텝 크기: 10
- 확장된 데이터: 0~4599 시점 (4600개 윈도우)

**데이터 구조**
```python
# 입력 데이터: (B, 50, 52)
# B: 배치 크기 (시뮬레이션 런 수)
# 50: 윈도우 크기
# 52: 센서 개수
```

## 파이프라인 동작 원리

### 1단계: Model1 (Fault 탐지 + 분류)
- **LSTM GENERATOR + CNN1D2D Discriminator** 사용
- 슬라이딩 윈도우 기반 배치 처리
- Fault 시점 탐지 (0~4599 → 0~959 변환)
- Fault 종류 분류 (21가지)
- 정상 상태 감지 시 파이프라인 조기 종료

### 2단계: Model2 (조작 변수 정상화) - 추후 추가 예정
- KNN?? 사용
- 조작 변수만 추출하여 정상화
- Fault 정보를 조건으로 사용

### 3단계: Model3 (반응 변수 예측) - 추후 추가 예정
- ?? 사용
- 정상화된 조작 변수를 기반으로 반응 변수 예측
- Fault 정보를 조건으로 사용

### 4단계: Model4 (정상 여부 재분류)
- **Model1 재사용**
- 정상화된 데이터 (m' + x')를 입력으로 사용
- 정상 분류 시: 정상화 완료, 파이프라인 종료
- 비정상 분류 시: Model2로 반복 (최대 3회)

## 사용 방법

### 1. 환경 설정

```bash
# 1. 새 CONDA 환경 생성
conda create -n tep_project python=3.7 -y
conda activate tep_project

# 2. 기본 과학 계산 패키지 설치
conda install numpy pandas scipy matplotlib scikit-learn -y

# 3. PyTorch 설치
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

# 4. 기타 필수 패키지 설치
pip install pyreadr tensorboardx python-dotenv memory-profiler click pillow opencv-python scikit-image

# 5. 프로젝트 로컬 설치
pip install -e .
```

### 2. 데이터 준비

**필수: NPY 데이터 파일**

```bash
data/
├── final_X.npy  # 전처리된 입력 데이터
└── final_Y.npy  # 전처리된 라벨 데이터
```

### 3. 파이프라인 실행

**메인 파이프라인 실행**

```bash
# 전체 파이프라인 실행
python src/main/pipeline.py
```

**파이프라인 동작 과정**
1. Model1: Fault 탐지 및 분류
2. 정상 상태 → 파이프라인 종료
3. 비정상 상태 → Model2, Model3, Model4 실행
4. Model4 결과에 따라 반복 또는 종료

## Model1 상세 정보

### CNN1D2D Discriminator 구조
- **1D Convolution**: 시계열 패턴 학습
- **2D Convolution**: 센서 간 상관관계 학습
- **Multitask Learning**: 결함 분류 + 실제/가짜 판별
- **배치 처리**: 슬라이딩 윈도우 기반

### 슬라이딩 윈도우 변환
```python
# 슬라이딩 윈도우 시점 → 원본 시점 변환
window_num = window_index // window_size
timestep_in_window = window_index % window_size
original_time = window_num * step_size + timestep_in_window
```

### 결과 출력
- **파이프라인 모델들**: 슬라이딩 윈도우 인덱스 (0~4599)
- **LLM**: 원본 시점 (0~959)

## 향후 개발 계획

### Model2 (조작 변수 정상화)
- Conditional TCN-AE 구현
- 조작 변수 (11개) 추출 및 정상화
- Fault 정보를 조건으로 사용

### Model3 (반응 변수 예측)
- RSSM (Recurrent State Space Model) 구현
- 정상화된 조작 변수 기반 반응 변수 예측
- 시계열 예측 모델

### LLM 통합
- Gemini API 연동
- 결과 해설 및 분석
- 사용자 친화적 출력

### 파이프라인 실행 (미완)

```bash
# 전체 파이프라인 테스트
python src/main/pipeline.py
```