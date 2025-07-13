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

    def normalize_by_knn(self, fault_seq: torch.Tensor, normal_db_part: torch.Tensor, k: int = 3, method: str = 'mean'):
        """
        단일 시퀀스에 대해 KNN 기반 보정 수행
        Args:
            fault_seq: (T_part, D)
            normal_db_part: (N_normal, T_part, D)  # 정상 DB 고장 이후 부분
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
                              fault_time: int,
                              k: int = 3,
                              method: str = 'mean') -> Dict[str, np.ndarray]:
        """
        Model1 출력을 받아 고장 이후 구간의 조작 변수(M)만 보정 (Model3과 동일한 fault_time 처리)

        Args:
            model1_output: (N, T, D) - Model1 출력 (원본과 동일)
            fault_time: 고장 시점 인덱스 (슬라이딩 윈도우 인덱스)
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
        
        # X/M 분할 (반응 변수: 0~40, 조작 변수: 41~51)
        x_part = input_tensor[:, :, :41]  # (N, T, 41) - 반응 변수 (그대로 유지)
        m_part = input_tensor[:, :, 41:]  # (N, T, 11) - 조작 변수 (보정 대상)
        
        output_m = m_part.clone()
        
        # fault_time을 배치 인덱스로 변환 (Model3과 동일)
        batch_idx = fault_time // 50  # 어느 배치에 속하는지
        timestep_in_batch = fault_time % 50  # 배치 내 시점
        
        print(f"fault_time: {fault_time} -> 배치 {batch_idx}, 배치 내 시점 {timestep_in_batch}")
        
        for i in range(N):
            if fault_time is None or fault_time >= T:
                continue
                
            # Model3과 동일한 방식: 배치 단위로 처리
            if batch_idx < m_part.shape[0]:  # 배치 범위 내
                # 컨텍스트: 배치 0~batch_idx (현재 배치까지 포함)
                context_m = m_part[i, :batch_idx+1, :]  # 이전 배치들 + 현재 배치
                
                # 미래: 배치 batch_idx+1부터 끝까지
                if batch_idx + 1 < m_part.shape[0]:
                    post_fault_m = m_part[i, batch_idx+1:, :]  # 이후 배치들
                else:
                    # 현재 배치가 마지막인 경우 빈 텐서 생성
                    post_fault_m = torch.empty(0, m_part.shape[1], m_part.shape[2], device=m_part.device)
                
                # 정상 DB도 동일한 방식으로 처리
                if batch_idx < self.normal_db.shape[0]:
                    normal_db_context = self.normal_db[:, :batch_idx+1, 41:]  # (N_normal, batch_idx+1, 11)
                    normal_db_post_fault = self.normal_db[:, batch_idx+1:, 41:]  # (N_normal, T-batch_idx-1, 11)
                else:
                    normal_db_context = self.normal_db[:, :, 41:]  # (N_normal, T, 11)
                    normal_db_post_fault = torch.empty(0, m_part.shape[1], m_part.shape[2], device=m_part.device)
                
                # KNN 보정 (조작 변수만) - Model3과 동일한 방식
                if post_fault_m.shape[0] > 0:
                    corrected_m = self.normalize_by_knn(post_fault_m, normal_db_post_fault, k=k, method=method)  # (T-batch_idx-1, 11)
                    output_m[i, batch_idx+1:, :] = corrected_m
            else:
                # fault_time이 범위를 벗어난 경우 기본값 사용
                context_m = m_part[i, :, :]  # 전체 시퀀스
                post_fault_m = torch.empty(0, m_part.shape[1], m_part.shape[2], device=m_part.device)

        # X와 보정된 M 결합
        combined_data = torch.cat([x_part, output_m], dim=2)  # (N, T, 52)
        
        # 변화량 요약 저장 (조작 변수만)
        delta_summary = self.compute_delta_summary(
            before=model1_output[:, fault_time:, 41:],  # 조작 변수만
            after=combined_data.detach().cpu().numpy()[:, fault_time:, 41:]  # 조작 변수만
        )

        self.results = {
            'fault_time': fault_time,
            'batch_idx': batch_idx,
            'timestep_in_batch': timestep_in_batch,
            'k_used': k,
            'normalized': True,
            'normalized_m_only': True,
            'delta_summary': delta_summary
        }

        reconstructed_all = combined_data.detach().cpu().numpy()
        x_part_np = reconstructed_all[:, :, :41]  # (N, T, 41)
        m_part_np = reconstructed_all[:, :, 41:]  # (N, T, 11)

        return {
            'reconstructed_all': reconstructed_all,
            'X': x_part_np,
            'M': m_part_np
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