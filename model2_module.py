import numpy as np
import torch
from typing import Dict, Tuple, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model2Module:
    """
    Model2: 순수 KNN 기반 보정기
    - 입력: Model1 출력 및 원본 시퀀스, 고장 시점
    - 출력: Model3 입력용 보정 시퀀스 (N, T, D) 및 X/M 분할
    """

    def __init__(self, normal_db: np.ndarray):
        """
        Args:
            normal_db: 정상 DB, shape (N_normal, T, D)
        """
        self.normal_db = torch.tensor(normal_db, dtype=torch.float32).to(DEVICE)  # (N_normal, T, D)
        self.T = normal_db.shape[1]
        self.D = normal_db.shape[2]
        self.results = {}  # 내부 결과 저장용

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

    def normalize_by_knn(self, fault_seq: torch.Tensor, k: int = 3, method: str = 'mean') -> torch.Tensor:
        """
        단일 시퀀스에 대해 KNN 기반 보정 수행
        Args:
            fault_seq: (T, D) - 고장 이후 구간
        Returns:
            normalized_seq: (T, D)
        """
        distances = self.compute_mse(self.normal_db, fault_seq)  # (N_normal,)
        topk_idx = torch.topk(distances, k=k, largest=False).indices  # (k,)
        topk_seqs = self.normal_db[topk_idx]  # (k, T, D)

        if method == 'mean':
            corrected = torch.mean(topk_seqs, dim=0)  # (T, D)
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
                              method: str = 'mean') -> Dict[str, np.ndarray]:
        """
        Model1 출력 + 원본을 받아 고장 이후 구간만 보정

        Args:
            model1_output: (N, T, D) - Model1 출력
            original_input: (N, T, D) - 원본 입력
            fault_time: 고장 시점 인덱스
            k: 최근접 이웃 개수
            method: 'mean' 또는 'first'

        Returns:
            {
                'reconstructed_all': (N, T, D),
                'X': (N, T, 41),
                'M': (N, T, 11)
            }
        """
        N, T, D = model1_output.shape
        assert D == 52, "전체 시퀀스는 52차원 (반응 + 조작 변수) 이어야 합니다."

        input_tensor = torch.tensor(model1_output, dtype=torch.float32).to(DEVICE)  # (N, T, D)
        orig_tensor = torch.tensor(original_input, dtype=torch.float32).to(DEVICE)

        output_tensor = input_tensor.clone()

        for i in range(N):
            if fault_time is None or fault_time >= T:
                continue
            # 고장 이후 부분만 보정
            post_fault = input_tensor[i, fault_time:, :]
            corrected = self.normalize_by_knn(post_fault, k=k, method=method)  # (T-fault_time, D)
            output_tensor[i, fault_time:, :] = corrected

        # X/M 분할 (반응 변수: 0~40, 조작 변수: 41~51)
        reconstructed_all = output_tensor.detach().cpu().numpy()
        x_part = reconstructed_all[:, :, :41]  # (N, T, 41)
        m_part = reconstructed_all[:, :, 41:]  # (N, T, 11)

        # 변화량 요약 저장
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

        return {
            'reconstructed_all': reconstructed_all,
            'X': x_part,
            'M': m_part
        }

    def compute_delta_summary(self, before: np.ndarray, after: np.ndarray, topk: int = 3):
        """
        변수별 보정 전후 변화량 통계 요약
        """
        delta = np.abs(after - before).mean(axis=(0, 1))  # (D,)
        topk_idx = np.argsort(delta)[-topk:][::-1].tolist()
        return {
            'topk_variables': topk_idx,
            'mean_deltas': delta[topk_idx].tolist()
        }

    def get_results_for_llm(self) -> Dict[str, Any]:
        """
        LLM에 전달할 보정 결과 요약 (Model2 기준)
        """
        result = {
            'fault_time': self.results.get('fault_time'),
            'k_used': self.results.get('k_used'),
            'normalized': self.results.get('normalized', False)
        }

        # optional: 변화량 큰 변수 요약 포함
        if 'delta_summary' in self.results:
            result['delta_summary'] = self.results['delta_summary']

        return result