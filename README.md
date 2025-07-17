# Tennessee Eastman Process (TEP) 공정 이상 탐지 및 정상화 파이프라인

## 팀 정보
**2025 AI 커리어패스 프로그램 - 팀3**
- 국민대학교 박규민
- 동양미래대학교 방석영
- 세종대학교 엄태호
- 한양여자대학교 조유영

## 프로젝트 개요

**4단계 공정 이상 분석 및 정상화 시스템**
- Model1: Fault 시점 탐지 + Fault 종류 분류 (LSTM Generator + CNN1D2D Discriminator)
- Model2: 조작 변수 정상화 (KNN 기반)
- Model3: 반응 변수 예측 (TCNSeq2Seq)
- Model4: 정상 여부 재분류 (Model1 재사용)

**주요 특징**
- 52개 센서 데이터 기반 시계열 분석
- 13가지 결함 유형 분류 (정상상태 포함)
- 슬라이딩 윈도우 기반 배치 처리 (50, 10)
- 반복 정상화 파이프라인 (최대 3회)
- LLM 기반 결과 해설

## 프로젝트 구조

```
Team-3---AI-Career-Path/
├── app.py
├── model_pretrained/
│   ├── model1/
│   │   ├── 30_epoch_checkpoint.pth
│   └── model3/
│       └── 30_epoch_checkpoint.pth
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── LLM/
│   │   ├── __init__.py
│   │   ├── LLM.py
│   │   └── prompts/
│   │       ├── base_prompt.txt
│   │       └── dataset_prompt.txt
│   │       └── fault_list.txt
│   ├── main/
│   │   ├── __init__.py
│   │   ├── model1_module.py
│   │   ├── model2_module.py
│   │   ├── model3_module.py
│   │   ├── pipeline.py
│   │   └── pipeline_no_model2.py
│   ├── model1/
│   │   ├── __init__.py
│   │   ├── convolutional_models.py
│   │   ├── evaluate_model.py
│   │   ├── recurrent_models.py
│   │   ├── train_model.py
│   │   └── utils.py
│   └── model3/
│       ├── evaluate_model3.py
│       ├── model3.py
│       └── train_model3.py
├── test_per_fault/
│   ├── fault_00_X.npy
│   ├── fault_00_Y.npy
│   ├── fault_01_X.npy
│   ├── fault_01_Y.npy
│   ├── fault_02_X.npy
│   ├── fault_02_Y.npy
│   ├── fault_03_X.npy
│   ├── fault_03_Y.npy
│   ├── fault_04_X.npy
│   ├── fault_04_Y.npy
│   ├── fault_05_X.npy
│   ├── fault_05_Y.npy
│   ├── fault_06_X.npy
│   ├── fault_06_Y.npy
│   ├── fault_07_X.npy
│   ├── fault_07_Y.npy
│   ├── fault_08_X.npy
│   ├── fault_08_Y.npy
│   ├── fault_09_X.npy
│   ├── fault_09_Y.npy
│   ├── fault_10_X.npy
│   ├── fault_10_Y.npy
│   ├── fault_11_X.npy
│   ├── fault_11_Y.npy
│   ├── fault_12_X.npy
│   └── fault_12_Y.npy
```

## 데이터셋 정보

**Tennessee Eastman Process 데이터**
- 센서 개수: 52개 (22개 공정 측정값, 19개 분석 측정값, 11개 조작 변수)
- 결함 유형: 13가지 (정상상태 포함)
- 샘플링 주기: 3분

**슬라이딩 윈도우 처리**
- 원본 데이터: 0~959 시점 (960개)
- 윈도우 크기: 50
- 스텝 크기: 10
- 확장된 데이터: 0~4599 시점 (92개 윈도우)

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
- Fault 시점 탐지
- Fault 종류 분류 (13가지)

### 2단계: Model2 (조작 변수 정상화)
- **KNN 기반 보정** 사용
- 고장 시점 이후 구간만 정상화
- 정상 DB와 유사한 패턴으로 보정

### 3단계: Model3 (반응 변수 예측)
- **TCNSeq2Seq** 사용
- 정상화된 조작 변수를 기반으로 반응 변수 예측
- 시계열 예측 모델

### 4단계: Model4 (정상 여부 재분류)
- **Model1 재사용**
- 정상화된 데이터를 입력으로 사용
- 정상 분류 시: 정상화 완료, 파이프라인 종료
- 비정상 분류 시: Model2로 반복 (최대 3회)

## 사용 방법

### 1. 환경 설정 및 패키지 설치

```bash
# 1. 새 가상환경 생성 (python 3.10)
python -m venv tep_env

# 2. 가상환경 활성화
# (Windows)
tep_env\Scripts\activate

# (Linux/Mac)
source tep_env/bin/activate

# 3. 필수 패키지 설치
pip install -r requirements.txt

# 4. 프로젝트 로컬 설치 (editable mode)
pip install -e .
```

### 2. 파이프라인 실행

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

### 3. Model1 학습 방법

**Model1(GAN 기반 Fault 탐지/분류) 학습**

```bash
python -m src.model1.train_model
```
- 기타 옵션은 `python -m src.model1.train_model --help`로 확인

### 4. Model3 학습 방법

**Model3(TCNSeq2Seq 기반 반응 변수 예측) 학습**

```bash
python -m src.model3.train_model3
```
- 기타 옵션은 `python -m src.model3.train_model3 --help`로 확인

## Model1 상세 정보

### GAN 구조: LSTM Generator + CNN1D2D Discriminator
- **LSTM Generator**
  - 정상/비정상 시계열 데이터를 생성
  - Discriminator가 더 강력하게 학습될 수 있도록 다양한 시나리오의 데이터를 만듦
- **CNN1D2D Discriminator**
  - 입력 시계열이 실제인지(Real/Fake) 판별
  - 동시에 fault 종류(13가지) 분류
  - 1D Convolution: 시계열 패턴 학습
  - 2D Convolution: 센서 간 상관관계 학습
  - Multitask Learning: 결함 분류 + 실제/가짜 판별

> LSTM Generator와 Discriminator가 경쟁적으로 학습(GAN 구조)하여,  
> Discriminator가 더 정교하게 fault를 탐지/분류할 수 있도록 Generator가 다양한 데이터를 생성해줍니다.

## Model2 상세 정보

### KNN 기반 조작 변수(m) 정상화
- **정상 DB 활용**: 유사한 패턴의 정상 데이터로 조작 변수(m, 11개) 보정
- **고장 시점 이후만 보정**: 고장 이전 데이터는 유지, 이후 구간의 m만 정상화
- **거리 기반 선택**: MSE 거리로 가장 유사한 k개 정상 시퀀스 선택
- **평균화 보정**: 선택된 k개 시퀀스의 m값 평균으로 보정
- **반응 변수(x, 41개)는 이 단계에서 직접 변경하지 않음**

## Model3 상세 정보

### TCNSeq2Seq 기반 반응 변수(x) 예측
- **Temporal Convolutional Network**: 시계열 패턴 학습
- **Sequence-to-Sequence**: 입력 시퀀스를 출력 시퀀스로 변환
- **보정된 조작 변수(m) 기반**: Model2에서 정상화된 m(11개)을 입력으로 받아, 그에 따라 변화할 반응 변수(x, 41개)를 시계열 예측
- **고장 시점 이후만 예측**: 고장 이전의 x(반응 변수)는 유지, 이후 구간의 x만 예측하여 업데이트
- 즉, "정상화된 m이 실제로 적용된다면 x가 어떻게 변할지"를 예측하는 단계

## 파이프라인 실행 예시

```bash
# 전체 파이프라인 테스트
python -m src.main.pipeline

# Model2 미포함 파이프라인 테스트
python -m src.main.pipeline_no_model2
```
- 위 명령어는 Model2(조작 변수 정상화) 단계를 생략하고, Model1 → Model3 → Model4만 실행하는 간소화 버전 파이프라인입니다.
- 정상화 없이 비정상 데이터를 바로 예측 및 재분류하는 실험/비교용으로 사용할 수 있습니다.

## 스트림릿 웹 애플리케이션

### 접속 주소
**https://tep-process-anomaly-detection-and-normalization-pipeline.streamlit.app/**

### 주요 기능

#### 1. **실시간 공정 데이터 분석**
- 52개 센서 데이터 시각화
- 시계열 그래프로 공정 상태 모니터링
- 실시간 이상 탐지 결과 표시

#### 2. **4단계 파이프라인 실행**
- **Model1**: Fault 시점 탐지 및 종류 분류
- **Model2**: 조작 변수 정상화
- **Model3**: 반응 변수 예측
- **Model4**: 정상화 성공 여부 재분류

#### 3. **인터랙티브 결과 시각화**
- 각 단계별 결과 그래프
- 정상화 전후 비교 차트
- 센서별 상세 분석

#### 4. **LLM 기반 결과 해설**
- AI 전문가가 분석 과정 설명
- 비전문가도 이해하기 쉬운 설명
- 단계별 추론 과정 상세 서술

#### 5. **사용자 친화적 인터페이스**
- 직관적인 웹 인터페이스
- 드래그 앤 드롭 파일 업로드
- 실시간 처리 상태 표시

### 제공되는 분석 정보

#### **Model1 결과**
- Fault 발생 시점 탐지
- 13가지 결함 유형 분류 (정상상태 포함)
- 원본 데이터 이상 탐지 결과

#### **Model2 결과**
- 조작 변수 정상화 결과
- 정상화된 조작 변수(m') 시퀀스
- KNN 기반 보정 과정

#### **Model3 결과**
- 반응 변수 예측 결과
- 예측된 반응 변수(x') 시퀀스
- TCNSeq2Seq 모델 성능

#### **Model4 결과**
- 최종 분류 결과
- 정상화 성공 여부
- 시스템 안정성 판단