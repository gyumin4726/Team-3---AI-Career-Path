import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import TEPNPYDataset, CSVToTensor

# 데이터 로드 및 구조 확인
print("1. 원본 데이터 구조 확인")
train_x = np.load('data/train_X.npy')
train_y = np.load('data/train_Y.npy')
print(f"train_X.npy shape: {train_x.shape}")  # (N, 50, 52)
print(f"train_Y.npy shape: {train_y.shape}")  # (N, 13)

# 전체 시뮬레이션 수 계산
# 한 시뮬레이션은 500 시점이 있고, 윈도우 크기 50, 스트라이드 10으로 슬라이딩하면
# 한 시뮬레이션당 46개의 윈도우가 생성됨
total_windows = len(train_x)  # 전체 윈도우 수
n_simulations = total_windows // 46  # 시뮬레이션 수
print(f"\n2. 시뮬레이션 계산")
print(f"총 윈도우 수: {total_windows}")
print(f"시뮬레이션 수: {n_simulations} (윈도우 수 ÷ 46)")
print(f"마지막 시뮬레이션의 윈도우 수: {total_windows % 46}")

# 데이터셋 생성 (transform은 numpy → torch 변환만)
print("\n3. 데이터셋 테스트")
dataset = TEPNPYDataset(
    data_path='data/train_X.npy',
    labels_path='data/train_Y.npy',
    transform=CSVToTensor()
)

# 첫 번째 시뮬레이션의 윈도우들 확인
print("\n4. 첫 번째 시뮬레이션의 윈도우들 확인")
for i in range(50):  # 첫 50개 윈도우 확인
    sample = dataset[i]
    sim_idx = sample['sim_idx']
    label = torch.argmax(sample['label']).item()
    print(f"윈도우 {i:2d}: 시뮬레이션 {sim_idx}, 라벨 {label}")

# 시뮬레이션별 통계
print("\n5. 시뮬레이션별 윈도우 수 확인")
sim_windows = {}  # 시뮬레이션별 윈도우 수
sim_labels = {}   # 시뮬레이션별 라벨

for i in range(min(500, len(dataset))):  # 처음 500개 윈도우만 확인
    sample = dataset[i]
    sim_idx = sample['sim_idx']
    label = torch.argmax(sample['label']).item()
    
    if sim_idx not in sim_windows:
        sim_windows[sim_idx] = 0
        sim_labels[sim_idx] = label
    sim_windows[sim_idx] += 1

print("\n처음 5개 시뮬레이션 통계:")
for sim_idx in sorted(list(sim_windows.keys()))[:5]:
    print(f"시뮬레이션 {sim_idx}:")
    print(f"  - 윈도우 수: {sim_windows[sim_idx]}")
    print(f"  - 라벨: {sim_labels[sim_idx]}") 