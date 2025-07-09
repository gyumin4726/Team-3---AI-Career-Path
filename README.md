# Tennessee Eastman Process 결함 탐지 - Temporal Deep Learning 모델

## 팀 정보
**2025 AI 커리어패스 프로그램 - 팀3**
- 국민대학교 박규민
- 동양미래대학교 방석영
- 세종대학교 엄태호
- 한양여자대학교 조유영

## 프로젝트 개요

**핵심 모델: GAN v5**
- CNN1D2D + GAN 결합 아키텍처
- 멀티태스크 학습: 결함 분류 + 데이터 생성 + 실제/가짜 판별
- 시계열 패턴 학습: LSTM Generator + CNN1D2D Discriminator

**주요 특징**
- 52개 센서 데이터 기반 시계열 분석
- 21가지 결함 유형 분류 (정상상태 포함)
- 하이브리드 딥러닝 아키텍처 (CNN1D2D + GAN)
- Multitask Learning 접근법
- 데이터 생성 및 증강 기능

## 프로젝트 구조

```
tennessee_eastman_diploma/
├── src/
│   ├── data/              # 데이터 처리 및 로딩
│   │   ├── dataset.py     # TEP 데이터셋 클래스들
│   │   └── make_dataset.py
│   ├── models/            # 딥러닝 모델 구현
│   │   ├── convolutional_models.py    # TCN, CNN1D2D 모델
│   │   ├── recurrent_models.py        # LSTM 기반 모델
│   │   ├── train_model.py             # 기본 CNN 분류기 훈련
│   │   ├── train_model_gan_v5.py      # GAN v5 모델 훈련 (메인)
│   │   ├── evaluate_model_csv.py      # 모델 평가 스크립트
│   │   └── utils.py                   # 유틸리티 함수
│   └── visualization/     # 시각화
│       └── visualize.py   # 시각화 스크립트
├── data/                # 데이터셋 저장소
│   ├── raw/             # 원본 RData 파일 위치
│   │   ├── TEP_FaultFree_Training.RData
│   │   ├── TEP_Faulty_Training.RData  
│   │   ├── TEP_FaultFree_Testing.RData
│   │   └── TEP_Faulty_Testing.RData
│   ├── train_faults/    # 훈련용 CSV 파일들
│   │   ├── train_fault_0.csv  # 정상 운전 데이터
│   │   ├── train_fault_1.csv  # 결함 1 데이터
│   │   ├── train_fault_2.csv  # 결함 2 데이터
│   │   ├── ...
│   │   └── train_fault_12.csv # 결함 12 데이터
│   └── test_faults/     # 테스트용 CSV 파일들
│       ├── test_fault_0.csv   # 정상 운전 테스트 데이터
│       ├── test_fault_1.csv   # 결함 1 테스트 데이터
│       ├── test_fault_2.csv   # 결함 2 테스트 데이터
│       ├── ...
│       └── test_fault_12.csv  # 결함 12 테스트 데이터
├── models/              # 훈련된 모델 저장소
├── setup.py             # 프로젝트 설정 파일
```

## 데이터셋 정보

**Tennessee Eastman Process 데이터**
- 센서 개수: 52개 (22개 공정 측정값, 19개 분석 측정값, 11개 조작 변수)
- 결함 유형: 21가지 (정상상태 포함)
- 샘플링 주기: 3분

**데이터 구조**
```python
# 훈련 데이터: 21 × 500 × 500 = 5,250,000 샘플
# 테스트 데이터: 21 × 500 × 960 = 10,080,000 샘플
```

**결함 특성**
- 정상 운전: 처음 1시간 (20샘플)
- 결함 발생: 1시간 후부터 (21번째 샘플부터)
- 결함 유형: IDV(1) ~ IDV(20) + 정상상태(0)

## 사용 방법

### 1. 환경 설정

**CONDA 환경 구성**

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

**필수: CSV 데이터 파일**

GAN v5 모델 훈련에는 다음 CSV 파일들이 필요합니다:

```bash
# 훈련 데이터
data/train_faults/
├── train_fault_0.csv  # 정상 운전 (18MB)
├── train_fault_1.csv  # 결함 1 (18MB)
├── train_fault_2.csv  # 결함 2 (18MB)
├── ...
└── train_fault_12.csv # 결함 12 (18MB)

# 테스트 데이터
data/test_faults/
├── test_fault_0.csv   # 정상 운전 (34MB)
├── test_fault_1.csv   # 결함 1 (34MB)
├── test_fault_2.csv   # 결함 2 (34MB)
├── ...
└── test_fault_12.csv  # 결함 12 (34MB)
```

**CSV 파일 형식:**
- 각 파일: 100개 시뮬레이션 런
- 훈련: 런당 500 시점 (25시간)
- 테스트: 런당 960 시점 (48시간) 
- 센서: 52개 (xmeas_1~xmeas_41, xmv_1~xmv_11)


### 3. 모델 훈련

**메인 모델 훈련**

```bash
# CNN1D2D+GAN 하이브리드 모델
python -m src.models.train_model_gan_v5 --cuda 0 --run_tag main_model
```

이 하나의 명령으로 전체 시스템이 완성됩니다:
- 시계열 데이터 생성 (Generator)
- 결함 분류 (13개 클래스: 정상 + 12가지 결함)  
- 정상/비정상 판별
- 멀티태스크 학습

**주요 옵션:**
- `--cuda 0`: GPU 번호 (0번 GPU 사용, 필수 옵션)
- `--run_tag main_model`: 실험 태그 (로그 구분용)


## GAN v5 모델 구조 및 작동 원리

### 모델 아키텍처

**GAN v5는 전통적인 데이터 증강 방식이 아닌 적대적 학습(Adversarial Training)을 활용한 robust한 결함 분류 모델입니다.**

#### 구성 요소
1. **Generator (LSTMGenerator)**
   - 역할: 가짜 TEP 시계열 데이터 생성
   - 입력: 노이즈(100차원) + 결함 유형 라벨(1차원)
   - 출력: 52개 센서의 시계열 데이터 (500 시간 스텝)

2. **Discriminator (CNN1D2DDiscriminatorMultitask)**
   - 역할: 실제/가짜 구별 + 결함 분류 (멀티태스크)
   - 입력: 시계열 데이터 (실제 또는 가짜)
   - 출력: 
     - `type_logits`: 결함 유형 분류 (13개 클래스: 정상 + 12가지 결함)
     - `real_fake_logits`: 실제/가짜 확률

### 훈련 과정

#### 1. Discriminator 훈련
```python
# 실제 데이터로 훈련
real_inputs, fault_labels = data["shot"], data["label"]  # CSV 파일에서 로드
type_logits, fake_logits = netD(real_inputs, None)
errD_type_real = cross_entropy_criterion(type_logits, fault_labels)  # 결함 분류 학습
errD_real = binary_criterion(fake_logits, REAL_LABEL)  # 실제 데이터 판별

# 가짜 데이터로 훈련
fake_inputs = netG(noise, labels)
type_logits, fake_logits = netD(fake_inputs.detach(), None)
errD_fake = binary_criterion(fake_logits, FAKE_LABEL)  # 가짜 데이터 판별
```

#### 2. Generator 훈련
```python
# Generator가 Discriminator를 속이도록 훈련
type_logits, fake_logits = netD(fake_inputs, None)
errG = binary_criterion(fake_logits, REAL_LABEL)  # 가짜를 진짜로 분류하도록

# 실제 데이터와 유사성 추구
errG_similarity = similarity(generated_data, real_inputs)
```


### 버전별 모델 조합

| 버전 | Generator | Discriminator | 특징 | 사용 목적 |
|------|-----------|---------------|------|----------|
| **GAN v5** | LSTM | CNN1D2D | 하이브리드 멀티태스크 | **메인 모델** (최고 성능) |
| **GAN v4** | LSTM | CausalConv | 멀티태스크 | 성능 비교 |
| **GAN v3** | CausalConv | CausalConv | 순수 CNN GAN | 실험/비교 |
| **GAN v2** | ❌ | TEPRNN | 순수 RNN | 베이스라인 |
| **베이스라인** | ❌ | CausalConv | 단순 분류 | 베이스라인 |


## 모델 평가

### GAN v5 모델 평가

학습이 완료된 후 테스트 데이터로 모델 성능을 평가할 수 있습니다.

#### 사용법
```bash
# 학습 완료된 discriminator 모델 평가 (결함 분류)
python -m src.models.evaluate_model_csv --model_path models/5_main_model/weights/199_epoch_discriminator.pth --cuda 0
```

#### 파라미터 설명
- `--cuda`: 사용할 GPU 번호 (0, 1, 2, ...)
- `--model_path`: 학습된 discriminator 모델 파일 경로
- `--random_seed`: 랜덤 시드 (옵션)