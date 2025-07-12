import numpy as np
import torch
from typing import Dict, Tuple
from typing import Union
from sklearn.decomposition import PCA

# ================================
# 설정값
# ================================
K = 3  # Nearest Neighbors 수
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# ================================
# Step 1: Normal DB 구성
# ================================
def build_normal_database(train_X, train_Y):
    normal_mask = (train_Y == 0)
    normal_db = torch.tensor(train_X[normal_mask], dtype=torch.float32).to(DEVICE)
    return normal_db

# ================================
# Step 2: 거리 계산 함수
# ================================
def compute_distances(fault_input, normal_db):
    diff = normal_db - fault_input.unsqueeze(0)  # (N_normal, 50, 11)
    distances = torch.mean(diff ** 2, dim=(1, 2))  # (N_normal,)
    return distances

# ================================
# Step 3: 보정 함수
# ================================
def normalize_by_knn(fault_input, normal_db, k=3, method='mean'):
    distances = compute_distances(fault_input, normal_db)
    topk_indices = torch.topk(distances, k=k, largest=False).indices  # k개 선택
    topk_seqs = normal_db[topk_indices]  # shape: (k, 50, 11)
    if method == 'mean':
        return torch.mean(topk_seqs, dim=0)
    elif method == 'first':
        return topk_seqs[0]
    else:
        raise ValueError("지원되지 않는 method입니다: mean / first 중 선택하세요.")

# ================================
# 전체 파이프라인 예시
# ================================
def knn_normalize_batch(test_X, normal_db, k=3, method='mean'):
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(DEVICE)
    normalized_list = []
    for i in range(test_X_tensor.shape[0]):
        normalized = normalize_by_knn(test_X_tensor[i], normal_db, k=k, method=method)
        normalized_list.append(normalized.cpu().numpy())
    return np.stack(normalized_list, axis=0)

class Model2Module:
    """
    Model2: 고장 시점 이후 정상화 (KNN 기반, MSE 사용)
    + Fault별 정상-고장 평균 거리 분석 지원
    """

    def __init__(self, normal_db: np.ndarray):
        """
        Args:
            normal_db: (N_normal, 50, 11) 형태의 정상 시퀀스 DB (np.ndarray 또는 tensor 허용)
        """
        if isinstance(normal_db, np.ndarray):
            self.normal_db = torch.tensor(normal_db, dtype=torch.float32).to(DEVICE)
        elif isinstance(normal_db, torch.Tensor):
            self.normal_db = normal_db.to(DEVICE)
        else:
            raise TypeError("normal_db는 numpy.ndarray 또는 torch.Tensor 여야 합니다.")

    def normalize_after_fault(self,
                              input_sequence: np.ndarray,
                              fault_time: int,
                              k: int = 3,
                              method: str = 'mean') -> Tuple[np.ndarray, Dict]:
        if fault_time is None or fault_time >= input_sequence.shape[0]:
            print("[Model2] 고장 시점이 유효하지 않습니다. 입력 그대로 반환합니다.")
            return input_sequence, {'normalized': False}

        post_fault_input = input_sequence[fault_time:]
        normalized = knn_normalize_batch(
            post_fault_input,
            self.normal_db,
            k=k,
            method=method
        )

        full_output = np.concatenate([
            input_sequence[:fault_time],
            normalized
        ], axis=0)

        return full_output, {
            'normalized': True,
            'fault_time': fault_time,
            'num_normalized_windows': len(post_fault_input)
        }

    def analyze_fault_distance(self, 
                           fault_data: np.ndarray,
                           normal_data: np.ndarray = None) -> float:
        """
        Fault 평균 시퀀스와 Normal 평균 시퀀스 간의 MSE 거리 계산
        """
        if normal_data is None:
            normal_data = self.normal_db.cpu().numpy()  # 🔧 수정: Tensor → numpy

        mean_fault = np.mean(fault_data, axis=0)   # (50, 11)
        mean_normal = np.mean(normal_data, axis=0) # (50, 11)

        mse = np.mean((mean_fault - mean_normal) ** 2)
        return mse


    def analyze_all_faults(self, fault_db: Dict[int, np.ndarray]) -> Dict[int, float]:
        results = {}
        for fault, data in fault_db.items():
            dist = self.analyze_fault_distance(data)
            results[fault] = dist
        return results
    
    
class PCABasedNormalizer:
    def __init__(self, normal_db: np.ndarray, n_components: int = 5):
        """
        PCA 기반 보정기 초기화
        Args:
            normal_db (np.ndarray): 정상 DB, shape (N, T, D)
            n_components (int): 차원 축소할 주성분 수
        """
        N, T, D = normal_db.shape
        self.n_components = n_components
        self.T = T
        self.D = D

        # 각 시점별 PCA 수행
        self.pca_models = []
        self.normal_proj = []
        for t in range(T):
            pca = PCA(n_components=n_components)
            X_t = normal_db[:, t, :]
            pca.fit(X_t)
            proj = pca.transform(X_t)
            self.pca_models.append(pca)
            self.normal_proj.append(proj)
        self.normal_proj = np.array(self.normal_proj)  # (T, N, n_components)

    def project(self, seq: np.ndarray) -> np.ndarray:
        """
        시퀀스를 PCA 공간으로 투영
        Args:
            seq (np.ndarray): shape (T, D)
        Returns:
            projected (np.ndarray): shape (T, n_components)
        """
        projected = np.zeros((self.T, self.n_components))
        for t in range(self.T):
            projected[t] = self.pca_models[t].transform(seq[t].reshape(1, -1))[0]
        return projected

    def compute_distances(self, projected_seq: np.ndarray) -> np.ndarray:
        """
        PCA 투영 공간에서 거리 계산 (MSE)
        Args:
            projected_seq (np.ndarray): shape (T, n_components)
        Returns:
            distances (np.ndarray): shape (N,)
        """
        diff = self.normal_proj - projected_seq[None, :, :]  # (N, T, n_components)
        distances = np.mean(diff ** 2, axis=(1, 2))  # (N,)
        return distances
    
    def knn_normalize(self,
                      fault_sequence: np.ndarray,
                      k: int = 3,
                      method: str = 'mean',
                      return_pca_vector: bool = False
                      ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert fault_sequence.shape == (self.T, self.D)

        normalized_sequence = []
        corrected_pca_sequence = []

        for t in range(self.T):
            pca = self.pca_models[t]
            normal_proj_t = self.normal_proj[:, t, :]  # (N, n_components)
            fault_proj_t = pca.transform(fault_sequence[t].reshape(1, -1))  # (1, n_components)

            dists = np.linalg.norm(normal_proj_t - fault_proj_t, axis=1)
            topk_indices = np.argsort(dists)[:k]
            topk_proj = normal_proj_t[topk_indices]

            if method == 'mean':
                corrected_proj = np.mean(topk_proj, axis=0)
            elif method == 'first':
                corrected_proj = topk_proj[0]
            else:
                raise ValueError("지원되지 않는 method입니다: 'mean' 또는 'first'")

            reconstructed = pca.inverse_transform(corrected_proj.reshape(1, -1)).reshape(-1)

            normalized_sequence.append(reconstructed)
            corrected_pca_sequence.append(corrected_proj)

        normalized_sequence = np.stack(normalized_sequence, axis=0)
        corrected_pca_sequence = np.stack(corrected_pca_sequence, axis=0)

        if return_pca_vector:
            return normalized_sequence, corrected_pca_sequence
        else:
            return normalized_sequence