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

    def __init__(self, normal_m_db: np.ndarray = None):
        """
        Args:
            normal_m_db: 정상 조작 변수 DB, shape (N_normal, T, 11) - None이면 build_normal_db()로 구축
        """
        if normal_m_db is not None:
            self.normal_m_db = torch.tensor(normal_m_db, dtype=torch.float32).to(DEVICE)  # (N_normal, T, 11)
            self.T = normal_m_db.shape[1]
            self.M = normal_m_db.shape[2]  # 조작 변수 차원 (11)
        else:
            self.normal_m_db = None
            self.T = None
            self.M = 11  # 조작 변수 차원 (11)
        self.results = {}  # 내부 결과 저장용

    @staticmethod
    def build_normal_db(data_path: str, sample_size: int = 5000, random_seed: int = 42) -> np.ndarray:
        """
        정상 조작 변수 DB 구축
        
        Args:
            data_path: 통합 X+M 데이터 경로 (shape: N, 50, 52)
            sample_size: 샘플링할 정상 데이터 개수
            random_seed: 랜덤 시드
            
        Returns:
            normal_m_db: 정상 조작 변수 DB (sample_size, 50, 11)
        """
        print(f"정상 DB 구축 시작: {data_path}")
        
        # 1. 통합된 X+M 윈도우 데이터 로드
        train_X_full = np.load(data_path)  # shape: (N, 50, 52)
        print(f"로드된 데이터 shape: {train_X_full.shape}")
        
        # 2. 조작변수 M만 추출 (마지막 11차원)
        train_M = train_X_full[:, :, 41:]  # shape: (N, 50, 11)
        print(f"조작 변수 추출 shape: {train_M.shape}")
        
        # 3. 지정된 개수만큼 무작위 샘플링
        np.random.seed(random_seed)
        subset_idx = np.random.choice(train_M.shape[0], size=sample_size, replace=False)
        train_M_sampled = train_M[subset_idx]  # shape: (sample_size, 50, 11)
        
        print(f"정상 DB 구축 완료: {train_M_sampled.shape}")
        return train_M_sampled

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
            fault_seq: (T, D) - 1D 시계열
            normal_db_part: (N_normal * T, D) - 정상 DB 시계열들
        Returns:
            normalized_seq: (T, D)
        """
        # 정상 DB의 원래 형태를 고려하여 재구성
        # normal_db_part는 (N_normal * 50, 11) 형태
        # fault_seq는 (batch_count * 50, 11) 형태
        
        # fault_seq를 배치 단위로 분할하여 각각 처리
        batch_count = fault_seq.shape[0] // 50  # 50은 시계열 길이
        fault_seqs = fault_seq.reshape(batch_count, 50, fault_seq.shape[1])  # (batch_count, 50, 11)
        
        # 정상 DB도 배치 단위로 분할
        N_normal = normal_db_part.shape[0] // 50  # 50은 시계열 길이
        normal_db_batches = normal_db_part.reshape(N_normal, 50, normal_db_part.shape[1])  # (N_normal, 50, 11)
        
        corrected_seqs = []
        
        for i in range(batch_count):
            # 각 배치에 대해 KNN 수행
            fault_batch = fault_seqs[i]  # (50, 11)
            
            # fault_batch를 2D로 확장하여 각 정상 DB 배치와 비교
            fault_batch_expanded = fault_batch.unsqueeze(0)  # (1, 50, 11)
            
            distances = self.compute_mse(normal_db_batches, fault_batch_expanded.squeeze(0))  # (N_normal,)
            topk_idx = torch.topk(distances, k=k, largest=False).indices  # (k,)
            topk_seqs = normal_db_batches[topk_idx]  # (k, 50, 11)

            if method == 'mean':
                corrected_batch = torch.mean(topk_seqs, dim=0)  # (50, 11)
            elif method == 'first':
                corrected_batch = topk_seqs[0]
            else:
                raise ValueError("지원되지 않는 method입니다: mean / first")
            
            corrected_seqs.append(corrected_batch)
        
        # 모든 배치 결과를 연결
        corrected = torch.cat(corrected_seqs, dim=0)  # (batch_count * 50, 11)
        
        return corrected

    def normalize_after_fault(self,
                              model1_output: np.ndarray,
                              fault_time: int,
                              k: int = 3,
                              method: str = 'mean',
                              data_path: str = None) -> Dict[str, np.ndarray]:
        """
        Model1 출력을 받아 고장 이후 구간의 조작 변수(M)만 보정 (Model3과 동일한 fault_time 처리)

        Args:
            model1_output: (N, T, D) - Model1 출력 (원본과 동일)
            fault_time: 고장 시점 인덱스 (슬라이딩 윈도우 인덱스)
            k: 최근접 이웃 개수
            method: 'mean' 또는 'first'
            data_path: 정상 DB가 없을 때 사용할 데이터 경로

        Returns:
            {
                'reconstructed_all': (N, T, D),
                'X': (N, T, 41),
                'M': (N, T, 11)
            }
        """
        N, T, D = model1_output.shape
        assert D == 52, "전체 시퀀스는 52차원 (반응 + 조작 변수) 이어야 합니다."

        # 정상 DB가 없으면 자동으로 구축
        if self.normal_m_db is None:
            if data_path is None:
                raise ValueError("정상 DB가 없고 data_path도 제공되지 않았습니다.")
            print("정상 DB가 없어 자동으로 구축합니다...")
            normal_m_db = self.build_normal_db(data_path)
            self.normal_m_db = torch.tensor(normal_m_db, dtype=torch.float32).to(DEVICE)
            self.T = normal_m_db.shape[1]

        # Model3과 동일한 방식으로 처리
        batch_size = 92  # Model3과 동일한 배치 크기
        num_simulations = len(model1_output) // batch_size
        
        print(f"Model2: 조작 변수 보정")
        print(f"입력: 전체 데이터 시퀀스 형태={model1_output.shape}")
        print(f"Fault 시점: {fault_time} (슬라이딩 윈도우 인덱스)")
        print(f"처리할 시뮬레이션 수: {num_simulations}")
        print(f"시뮬레이션당 윈도우 수: {batch_size}")
        
        corrected_m_list = []
        
        for sim_idx in range(num_simulations):
            print(f"시뮬레이션 {sim_idx + 1}/{num_simulations} 처리 중...")
            
            # 현재 시뮬레이션 데이터 추출 (Model3과 동일)
            start_idx = sim_idx * batch_size
            end_idx = start_idx + batch_size
            sim_data = model1_output[start_idx:end_idx]  # (92, 50, 52)
            
            # 변수 분할 (Model3과 동일)
            x_data = sim_data[:, :, :41]  # (92, 50, 41) - 반응 변수
            m_data = sim_data[:, :, 41:]  # (92, 50, 11) - 조작 변수
            
            # fault_time을 배치 인덱스로 변환 (Model3과 동일)
            batch_idx = fault_time // 50
            
            # fault_time이 현재 시뮬레이션 범위 내인지 확인 (Model3과 동일)
            if batch_idx < len(sim_data):
                # 컨텍스트: 배치 0~batch_idx (현재 배치까지 포함) - Model3과 동일
                x_ctx = x_data[:batch_idx+1, :, :]  # 이전 배치들 + 현재 배치
                m_ctx = m_data[:batch_idx+1, :, :]
                
                # 미래: 배치 batch_idx+1부터 끝까지 - Model3과 동일
                if batch_idx + 1 < len(sim_data):
                    m_fut = m_data[batch_idx+1:, :, :]  # 이후 배치들
                else:
                    # 현재 배치가 마지막인 경우 빈 텐서 생성
                    m_fut = np.empty((0, m_data.shape[1], m_data.shape[2]))
                
                # KNN 보정 (조작 변수만) - Model3과 동일한 방식
                if m_fut.shape[0] > 0:
                    # 정상 DB는 전체 시계열 길이(50)를 가지고 있으므로, 시계열 단위로 처리
                    # m_fut를 시계열로 변환: (batch_count, 50, 11) -> (batch_count * 50, 11)
                    m_fut_reshaped = m_fut.reshape(-1, m_fut.shape[-1])  # (batch_count * 50, 11)
                    
                    # 정상 DB도 동일한 방식으로 처리
                    normal_db_reshaped = self.normal_m_db.reshape(-1, self.normal_m_db.shape[-1])  # (N_normal * 50, 11)
                    
                    # KNN 보정 수행
                    m_fut_tensor = torch.FloatTensor(m_fut_reshaped).to(DEVICE)
                    normal_db_tensor = torch.FloatTensor(normal_db_reshaped.numpy()).to(DEVICE)
                    
                    corrected_m_fut_reshaped = self.normalize_by_knn(m_fut_tensor, normal_db_tensor, k=k, method=method)
                    corrected_m_fut_reshaped = corrected_m_fut_reshaped.cpu().numpy()
                    
                    # 원래 형태로 복원: (batch_count * 50, 11) -> (batch_count, 50, 11)
                    corrected_m_fut = corrected_m_fut_reshaped.reshape(m_fut.shape)
                    
                    # 새로운 시퀀스 생성 (Model3과 동일)
                    # fault_time 이전: 원본 M
                    # fault_time 이후: 보정된 M'
                    
                    # 컨텍스트와 보정 결과를 연결
                    new_m_sequence = np.concatenate([m_ctx, corrected_m_fut], axis=0)  # (total_batches, 50, 11)
                else:
                    new_m_sequence = m_ctx  # 보정할 미래가 없는 경우
                
                # 원본 형태로 복원 (92, 50, 11) - Model3과 동일
                if new_m_sequence.shape[0] < 92:
                    # 부족한 배치는 마지막 배치로 채움
                    last_batch = new_m_sequence[-1:].repeat(92 - new_m_sequence.shape[0], axis=0)
                    new_m_sequence = np.concatenate([new_m_sequence, last_batch], axis=0)
                elif new_m_sequence.shape[0] > 92:
                    # 초과하는 배치는 제거
                    new_m_sequence = new_m_sequence[:92]
                
                # 새로운 데이터 생성 (X + M') - Model3과 동일
                new_sim_data = np.concatenate([x_data, new_m_sequence], axis=2)  # (92, 50, 52)
                
                corrected_m_list.append(new_sim_data)
            else:
                # fault_time이 범위를 벗어난 경우 원본 데이터 사용
                corrected_m_list.append(sim_data)
        
        # 결과 합치기 (Model3과 동일)
        corrected_data = np.concatenate(corrected_m_list, axis=0)
        
        # 변화량 요약 저장 (조작 변수만)
        delta_summary = self.compute_delta_summary(
            before=model1_output[:, :, 41:],  # 조작 변수만 (전체)
            after=corrected_data[:, :, 41:],  # 조작 변수만 (전체)
            fault_time=fault_time
        )

        self.results = {
            'fault_time': fault_time,
            'batch_idx': batch_idx,
            'k_used': k,
            'normalized': True,
            'normalized_m_only': True,
            'delta_summary': delta_summary,
            'normalized_sequence': corrected_data  # LLM용 결과 저장
        }

        reconstructed_all = corrected_data
        x_part_np = reconstructed_all[:, :, :41]  # (N, T, 41)
        m_part_np = reconstructed_all[:, :, 41:]  # (N, T, 11)

        print(f"결과: 보정된 전체 시퀀스 형태={reconstructed_all.shape}")
        return {
            'reconstructed_all': reconstructed_all,
            'X': x_part_np,
            'M': m_part_np
        }

    def compute_delta_summary(self, before: np.ndarray, after: np.ndarray, fault_time: int, topk: int = 3):
        """
        변수별 보정 전후 변화량 통계 요약 (Model3과 동일한 배치 처리 방식)
        """
        if before.size == 0 or after.size == 0:
            return {
                'topk_variables': [],
                'mean_deltas': []
            }
        
        # fault_time을 배치 인덱스로 변환 (Model3과 동일)
        batch_idx = fault_time // 50
        
        # Model3과 동일한 방식으로 처리
        batch_size = 92
        num_simulations = len(before) // batch_size
        
        all_deltas = []
        
        for sim_idx in range(num_simulations):
            # 현재 시뮬레이션 데이터 추출 (Model3과 동일)
            start_idx = sim_idx * batch_size
            end_idx = start_idx + batch_size
            sim_before = before[start_idx:end_idx]  # (92, 50, 11)
            sim_after = after[start_idx:end_idx]    # (92, 50, 11)
            
            # fault_time이 현재 시뮬레이션 범위 내인지 확인 (Model3과 동일)
            if batch_idx < len(sim_before):
                # 배치 batch_idx+1부터 끝까지 (고장 이후 구간) - Model3과 동일
                if batch_idx + 1 < len(sim_before):
                    before_post_fault = sim_before[batch_idx+1:, :, :]  # 이후 배치들
                    after_post_fault = sim_after[batch_idx+1:, :, :]    # 이후 배치들
                else:
                    # 현재 배치가 마지막인 경우 빈 배열
                    before_post_fault = np.empty((0, sim_before.shape[1], sim_before.shape[2]))
                    after_post_fault = np.empty((0, sim_after.shape[1], sim_after.shape[2]))
            else:
                # fault_time이 범위를 벗어난 경우 빈 배열
                before_post_fault = np.empty((0, sim_before.shape[1], sim_before.shape[2]))
                after_post_fault = np.empty((0, sim_after.shape[1], sim_after.shape[2]))
            
            # 변화량 계산 (고장 이후 구간만)
            if before_post_fault.size > 0 and after_post_fault.size > 0:
                delta = np.abs(after_post_fault - before_post_fault).mean(axis=(0, 1))  # (11,)
                all_deltas.append(delta)
        
        # 모든 시뮬레이션의 변화량을 평균
        if all_deltas:
            mean_delta = np.mean(all_deltas, axis=0)  # (11,)
            topk_idx = np.argsort(mean_delta)[-topk:][::-1].tolist()
            return {
                'topk_variables': topk_idx,
                'mean_deltas': mean_delta[topk_idx].tolist()
            }
        else:
            return {
                'topk_variables': [],
                'mean_deltas': []
            }

    def summarize_top3_m_changes(self, normalized_sequence: np.ndarray, original_sequence: np.ndarray, fault_time: int):
        """
        Model2 정상화 전후로 변화가 큰 조작 변수 Top 3와 통계 요약 반환 (Model3과 동일한 배치 처리 방식)
        Args:
            normalized_sequence: (B, 50, 52) - Model2 정상화 결과
            original_sequence: (B, 50, 52) - 원본 입력 데이터
            fault_time: 슬라이딩 윈도우 인덱스 기준 fault 시점
        Returns:
            {
                'top3_indices': [int, int, int],
                'stats': {
                    idx: {
                        'before_mean': float,
                        'after_mean': float,
                        'delta_mean': float,
                        'before_std': float,
                        'after_std': float,
                        'delta_max': float,
                        'delta_min': float
                    }, ...
                }
            }
        """
        # 조작 변수(M)만 추출 (41~51)
        norm_m = normalized_sequence[:, :, 41:]
        orig_m = original_sequence[:, :, 41:]

        # Model3과 동일한 방식으로 처리
        batch_size = 92
        num_simulations = len(norm_m) // batch_size
        
        all_deltas = []
        all_stats = []
        
        for sim_idx in range(num_simulations):
            # 현재 시뮬레이션 데이터 추출 (Model3과 동일)
            start_idx = sim_idx * batch_size
            end_idx = start_idx + batch_size
            sim_norm_m = norm_m[start_idx:end_idx]  # (92, 50, 11)
            sim_orig_m = orig_m[start_idx:end_idx]  # (92, 50, 11)
            
            # fault_time 이후 구간만 추출 (Model3과 동일)
            B, T, M = sim_norm_m.shape
            total_steps = B * T
            norm_m_flat = sim_norm_m.reshape(total_steps, M)
            orig_m_flat = sim_orig_m.reshape(total_steps, M)
            
            # fault_time 이후만
            norm_m_after = norm_m_flat[fault_time:]
            orig_m_after = orig_m_flat[fault_time:]
            
            # 변수별 변화량 (정상화 - 원본)의 절대값 평균
            if norm_m_after.size > 0 and orig_m_after.size > 0:
                delta = np.abs(norm_m_after - orig_m_after)
                delta_mean = delta.mean(axis=0)  # (11,)
                all_deltas.append(delta_mean)
                
                # 통계 계산
                for idx in range(11):
                    before = orig_m_flat[:fault_time, idx]
                    after_norm = norm_m_flat[fault_time:, idx]
                    after_orig = orig_m_flat[fault_time:, idx]
                    
                    if before.size > 0 and after_norm.size > 0:
                        all_stats.append({
                            'idx': idx + 41,  # 실제 변수 인덱스
                            'before_mean': float(np.mean(before)),
                            'after_mean': float(np.mean(after_norm)),
                            'delta_mean': float(np.mean(np.abs(after_norm - after_orig))),
                            'before_std': float(np.std(before)),
                            'after_std': float(np.std(after_norm)),
                            'delta_max': float(np.max(np.abs(after_norm - after_orig))),
                            'delta_min': float(np.min(np.abs(after_norm - after_orig)))
                        })

        # 모든 시뮬레이션의 변화량을 평균
        if all_deltas:
            mean_delta = np.mean(all_deltas, axis=0)  # (11,)
            top3_indices = np.argsort(mean_delta)[-3:][::-1].tolist()
            top3_indices = [idx + 41 for idx in top3_indices]  # 실제 변수 인덱스로 변환
            
            # 통계 요약
            stats = {}
            for idx in top3_indices:
                m_idx = idx - 41  # 조작 변수 내 인덱스
                sim_stats = [s for s in all_stats if s['idx'] == idx]
                if sim_stats:
                    # 모든 시뮬레이션의 평균
                    stats[idx] = {
                        'before_mean': float(np.mean([s['before_mean'] for s in sim_stats])),
                        'after_mean': float(np.mean([s['after_mean'] for s in sim_stats])),
                        'delta_mean': float(np.mean([s['delta_mean'] for s in sim_stats])),
                        'before_std': float(np.mean([s['before_std'] for s in sim_stats])),
                        'after_std': float(np.mean([s['after_std'] for s in sim_stats])),
                        'delta_max': float(np.max([s['delta_max'] for s in sim_stats])),
                        'delta_min': float(np.min([s['delta_min'] for s in sim_stats]))
                    }
        else:
            top3_indices = []
            stats = {}
            
        return {
            'top3_indices': top3_indices,
            'stats': stats
        }

    def get_results_for_llm(self, original_sequence: np.ndarray, fault_time: int) -> dict:
        """
        LLM에게 전달할 Model2 요약 결과 반환
        Args:
            original_sequence: (B, 50, 52) - 원본 입력 데이터
            fault_time: 슬라이딩 윈도우 인덱스 기준 fault 시점
        Returns:
            {
                'top3_indices': [...],
                'top3_stats': {...},
                'summary': str
            }
        """
        # 정상화된 데이터는 self.results에서 가져오기
        normalized_sequence = self.results.get('normalized_sequence', original_sequence)
        summary_data = self.summarize_top3_m_changes(normalized_sequence, original_sequence, fault_time)
        top3_indices = summary_data['top3_indices']
        stats = summary_data['stats']

        # 간단한 요약 설명 생성
        summary_lines = [
            f"fault_time={fault_time} 이후 정상화된 조작 변수(M)의 변화가 큰 Top 3 변수는 {top3_indices}입니다.",
        ]
        for idx in top3_indices:
            s = stats[idx]
            summary_lines.append(
                f"  - 변수 {idx}: fault 이전 평균={s['before_mean']:.3f}, fault 이후 정상화 평균={s['after_mean']:.3f}, 변화량 평균={s['delta_mean']:.3f}, 변화량 최대={s['delta_max']:.3f}"
            )
        summary = '\n'.join(summary_lines)

        return {
            'top3_indices': top3_indices,
            'top3_stats': stats,
            'summary': summary
        }