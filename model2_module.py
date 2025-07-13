import numpy as np
import torch
from typing import Dict, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model2Module:
    """
    Model2: 순수 KNN 기반 보정기
    - 입력: Model1 출력 및 원본 시퀀스, 고장 시점
    - 출력: 보정된 조작 변수 M 시퀀스 (N, T, 11)
    """

    def __init__(self, normal_db: np.ndarray):
        """
        Args:
            normal_db: 정상 DB, shape (N_normal, T, 11)  # 조작 변수 11개만
        """
        self.normal_db = torch.tensor(normal_db, dtype=torch.float32).to(DEVICE)  # (N_normal, T, 11)
        self.T = normal_db.shape[1]
        self.D = normal_db.shape[2]
        self.results = {}

    def compute_mse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        MSE 거리 계산: (T, D) 간
        Args:
            a: (N, T, D)
            b: (T, D)
        Returns:
            distances: (N,)
        """
        return torch.mean((a - b) ** 2, dim=(1, 2))  # (N,)

    def normalize_by_knn(self, fault_seq: torch.Tensor, normal_db_part: torch.Tensor,
                         k: int = 3, method: str = 'mean') -> torch.Tensor:
        """
        단일 시퀀스에 대해 KNN 기반 보정 수행
        Args:
            fault_seq: (T_part, D)
            normal_db_part: (N_normal, T_part, D)
        Returns:
            normalized_seq: (T_part, D)
        """
        distances = self.compute_mse(normal_db_part, fault_seq)  # (N_normal,)
        topk_idx = torch.topk(distances, k=k, largest=False).indices  # (k,)
        topk_seqs = normal_db_part[topk_idx]  # (k, T_part, D)

        if method == 'mean':
            corrected = torch.mean(topk_seqs, dim=0)  # (T_part, D)
        elif method == 'first':
            corrected = topk_seqs[0]
        else:
            raise ValueError("지원되지 않는 method입니다: mean / first")

        return corrected

    def normalize_after_fault(self,
                              model1_output: np.ndarray,
                              original_input: np.ndarray,
                              fault_time: int,
                              k: int = 3,
                              method: str = 'mean') -> Dict[str, Any]:
        """
        Model1 출력 + 원본을 받아 고장 이후 구간만 보정 (조작 변수 11개만)

        Args:
            model1_output: (N, T, 11) - Model1 출력 (조작 변수만)
            original_input: (N, T, 11) - 원본 입력 (조작 변수만)
            fault_time: 고장 시점 인덱스
            k: 최근접 이웃 개수
            method: 'mean' 또는 'first'

        Returns:
            {
                'M': (N, T, 11)  # 보정된 조작 변수 시퀀스
            }
        """
        N, T, D = model1_output.shape
        assert D == 11, "입력 데이터는 11차원 조작 변수만 포함해야 합니다."

        input_tensor = torch.tensor(model1_output, dtype=torch.float32).to(DEVICE)  # (N, T, 11)
        output_tensor = input_tensor.clone()

        for i in range(N):
            if fault_time is None or fault_time >= T:
                continue
            post_fault = input_tensor[i, fault_time:, :]  # (T-fault_time, 11)
            normal_db_post_fault = self.normal_db[:, fault_time:, :]  # (N_normal, T-fault_time, 11)

            corrected = self.normalize_by_knn(post_fault, normal_db_post_fault, k=k, method=method)  # (T-fault_time, 11)
            output_tensor[i, fault_time:, :] = corrected

        reconstructed_all = output_tensor.detach().cpu().numpy()

        delta_summary = self.compute_delta_summary(
            before=model1_output[:, fault_time:, :],
            after=reconstructed_all[:, fault_time:, :]
        )

        self.results = {
            'fault_time': fault_time,
            'k_used': k,
            'normalized': True,
            'delta_summary': delta_summary
        }

        return {'M': reconstructed_all}

    def compute_delta_summary(self, before: np.ndarray, after: np.ndarray, topk: int = 3):
        delta = np.abs(after - before).mean(axis=(0, 1))  # (11,)
        topk_idx = np.argsort(delta)[-topk:][::-1].tolist()
        return {
            'topk_variables': topk_idx,
            'mean_deltas': delta[topk_idx].tolist()
        }

    def get_results_for_llm(self) -> Dict[str, Any]:
        result = {
            'fault_time': self.results.get('fault_time'),
            'k_used': self.results.get('k_used'),
            'normalized': self.results.get('normalized', False)
        }
        if 'delta_summary' in self.results:
            result['delta_summary'] = self.results['delta_summary']
        return result