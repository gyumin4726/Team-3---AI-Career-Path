import numpy as np

# 테스트 데이터 로드
print('데이터 로드 중...')
test_x = np.load('data/test_X.npy')
test_y = np.load('data/test_intY.npy')

# 데이터 크기 확인
print('원본 크기:')
print(f'test_X.shape: {test_x.shape}')
print(f'test_intY.shape: {test_y.shape}')

# 원본 클래스 분포 확인
unique, counts = np.unique(test_y, return_counts=True)
print('\n원본 클래스 분포:')
for cls, cnt in zip(unique, counts):
    print(f'클래스 {cls}: {cnt}개 샘플')

# 각 클래스별로 100분의 1 샘플링
sampled_indices = []
for cls in unique:
    # 해당 클래스의 인덱스 찾기
    cls_indices = np.where(test_y == cls)[0]
    # 100분의 1 크기로 랜덤 샘플링
    sample_size = len(cls_indices) // 100
    if sample_size > 0:  # 샘플이 있는 경우만
        sampled = np.random.choice(cls_indices, size=sample_size, replace=False)
        sampled_indices.extend(sampled)

# 샘플링된 인덱스로 데이터 추출
sampled_indices = np.array(sampled_indices)
test_x_small = test_x[sampled_indices]
test_y_small = test_y[sampled_indices]

print('\n샘플링된 크기:')
print(f'test_X_small.shape: {test_x_small.shape}')
print(f'test_intY_small.shape: {test_y_small.shape}')

# 샘플링된 클래스 분포 확인
unique, counts = np.unique(test_y_small, return_counts=True)
print('\n샘플링된 클래스 분포:')
for cls, cnt in zip(unique, counts):
    print(f'클래스 {cls}: {cnt}개 샘플')

# 새 파일로 저장
print('\n파일 저장 중...')
np.save('data/test_X_small.npy', test_x_small)
np.save('data/test_intY_small.npy', test_y_small)
print('저장 완료: data/test_X_small.npy, data/test_intY_small.npy') 