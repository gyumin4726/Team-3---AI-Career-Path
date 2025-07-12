#!/usr/bin/env python3
"""
Tennessee Eastman Process Transformer 모델 평가 스크립트
학습된 Causal Transformer Autoencoder를 사용하여 이상 측정값(m) 시퀀스 평가 및 시각화
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import click
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

from src.data.dataset import TEPNPYDataset, CSVToTensor
from src.models.transformer_models import MTCNAutoencoder


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model(model_path, device):
    """학습된 모델 로드"""
    logger = logging.getLogger(__name__)
    logger.info(f"모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 체크포인트에서 모델 state dict 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 생성
    model = MTCNAutoencoder(
        m_dim=11,  # 측정값만 사용
        channels=[32, 64, 64],
        kernel_size=3
    ).to(device)
    
    # state dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"모델 로드 완료: {type(model).__name__}")
    return model


def flatten_seq(seq_tensor):
    """시퀀스 텐서를 2D로 평탄화"""
    # 입력: [B, T, D] → [B, T*D]
    return seq_tensor.reshape(seq_tensor.shape[0], -1)


def plot_embedding(orig_seqs, rec_seqs, labels, save_path=None):
    """이상 측정값 시퀀스와 재구성 결과의 저차원 임베딩 시각화
    
    Args:
        orig_seqs: 원본 시퀀스 [N, T, D]
        rec_seqs: 재구성된 시퀀스 [N, T, D]
        labels: Fault 라벨 [N]
        save_path: 저장 경로
    """
    # Flatten sequences
    orig_flat = flatten_seq(orig_seqs)  # [N, T*D]
    rec_flat = flatten_seq(rec_seqs)    # [N, T*D]
    
    # PCA 학습 및 투영
    pca = PCA(n_components=2)
    
    # 원본 데이터로 PCA 학습
    pca.fit(orig_flat)
    
    # 각각 투영
    orig_pca = pca.transform(orig_flat)  # [N, 2]
    rec_pca = pca.transform(rec_flat)    # [N, 2]
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # 각 fault 유형별로 다른 색상 사용
    unique_faults = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_faults)))
    
    # 각 Fault별 입력과 재구성 결과
    for fault, color in zip(unique_faults, colors):
        mask = labels == fault
        # 이상 입력
        plt.scatter(orig_pca[mask, 0], orig_pca[mask, 1],
                   label=f'Fault {fault} input', color=color, alpha=0.6)
        # 이상 재구성 (같은 색상, 다른 마커)
        plt.scatter(rec_pca[mask, 0], rec_pca[mask, 1],
                   label=f'Fault {fault} reshape', color=color, marker='^', alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('이상 데이터 재구성 결과 PCA 시각화\n(원형: 이상 입력, 삼각형: 정상화 시도 결과)')
    plt.xlabel('주성분 1')
    plt.ylabel('주성분 2')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_reconstruction(model, test_loader, device, save_dir):
    """모델의 재구성 성능 평가"""
    logger = logging.getLogger(__name__)
    model.eval()
    
    orig_seqs = []
    rec_seqs = []
    fault_labels = []
    
    total_mse = 0
    fault_mse = defaultdict(lambda: {'total': 0, 'count': 0})
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 측정값(m)만 사용 [B, 50, 11]
            m_seq = batch["shot"][:, :, :11].to(device)
            labels = batch["label"]  # [B]
            
            # 첫 배치의 shape 출력
            if i == 0:
                logger.info("\n데이터 shape 정보:")
                logger.info(f"입력 데이터 shape: {m_seq.shape}")  # [B, 50, 11]
                logger.info(f"라벨 shape: {labels.shape}")  # [B]
            
            # 재구성 [B, 50, 11]
            m_rec = model(m_seq)
            
            if i == 0:
                logger.info(f"재구성 데이터 shape: {m_rec.shape}")  # [B, 50, 11]
            
            # MSE 계산
            mse = torch.nn.functional.mse_loss(m_rec, m_seq, reduction='none')  # [B, 50, 11]
            mse = mse.mean(dim=(1,2))  # [B]
            
            # 전체 MSE
            total_mse += mse.sum().item()
            
            # Fault별 MSE 계산
            for i, (label, sample_mse) in enumerate(zip(labels, mse)):
                fault = label.item()
                fault_mse[fault]['total'] += sample_mse.item()
                fault_mse[fault]['count'] += 1
            
            # CPU로 이동하고 numpy로 변환
            m_seq = m_seq.cpu().numpy()
            m_rec = m_rec.cpu().numpy()
            
            orig_seqs.append(m_seq)
            rec_seqs.append(m_rec)
            fault_labels.extend(labels.numpy())
    
    # 시퀀스 연결
    orig_seqs = np.concatenate(orig_seqs, axis=0)
    rec_seqs = np.concatenate(rec_seqs, axis=0)
    fault_labels = np.array(fault_labels)
    
    logger.info("\n전체 데이터 shape:")
    logger.info(f"원본 시퀀스 shape: {orig_seqs.shape}")  # [N, 50, 11]
    logger.info(f"재구성 시퀀스 shape: {rec_seqs.shape}")  # [N, 50, 11]
    logger.info(f"라벨 shape: {fault_labels.shape}")  # [N]
    
    # 전체 평균 MSE 계산
    n_samples = len(fault_labels)
    avg_mse = total_mse / n_samples
    logger.info(f"\n전체 평균 MSE: {avg_mse:.6f}")
    
    # Fault별 평균 MSE 계산
    fault_avg_mse = {}
    logger.info("\nFault별 통계:")
    for fault in fault_mse:
        fault_avg = fault_mse[fault]['total'] / fault_mse[fault]['count']
        fault_avg_mse[int(fault)] = {
            'mse': float(fault_avg),
            'n_windows': fault_mse[fault]['count']
        }
        logger.info(f"Fault {fault} 평균 MSE: {fault_avg:.6f} "
                   f"(윈도우 수: {fault_mse[fault]['count']})")
    
    # PCA와 t-SNE 시각화
    logger.info("\n시각화 생성 중...")
    plot_embedding(orig_seqs, rec_seqs, fault_labels,
                  save_path=os.path.join(save_dir, "pca_visualization.png"))
    
    return {
        'total_mse': avg_mse,
        'fault_mse': fault_avg_mse,
        'total_windows': n_samples
    }


def save_results(results, save_dir):
    """평가 결과 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 결과를 JSON 형식으로 저장
    results_path = os.path.join(save_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)


class ConditionalTCNAEEvaluator:
    """Conditional TCN-AE 모델 평가기"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def calculate_metrics(self, 
                         original_m: torch.Tensor,
                         normalized_m: torch.Tensor,
                         fault_mask: torch.Tensor) -> Dict[str, float]:
        """
        정상화 성능 메트릭 계산
        
        Args:
            original_m: 원본 조작변수 (B, T, M)
            normalized_m: 정상화된 조작변수 (B, T, M)
            fault_mask: Fault 발생 구간 마스크 (B, T), True=fault 구간
            
        Returns:
            Dict[str, float]: 평가 메트릭
        """
        # Fault 구간만 선택
        original_fault = original_m[fault_mask]
        normalized_fault = normalized_m[fault_mask]
        
        # 기본 메트릭
        metrics = {
            "mse": float(mean_squared_error(original_fault, normalized_fault)),
            "mae": float(mean_absolute_error(original_fault, normalized_fault)),
            "rmse": float(np.sqrt(mean_squared_error(original_fault, normalized_fault)))
        }
        
        # 변수별 정규화 효과 (변동성 감소율)
        var_reduction = {}
        for i in range(original_m.size(-1)):
            orig_std = torch.std(original_fault[:, i]).item()
            norm_std = torch.std(normalized_fault[:, i]).item()
            reduction = (orig_std - norm_std) / orig_std * 100
            var_reduction[f"mv{i+1}_reduction"] = float(reduction)
        
        metrics.update(var_reduction)
        
        # 정상화 신뢰도 점수 (0~1)
        # 1에 가까울수록 정상 범위 내 값으로 정상화됨
        confidence = self._calculate_normalization_confidence(
            original_fault, normalized_fault
        )
        metrics["normalization_confidence"] = float(confidence)
        
        return metrics
    
    def _calculate_normalization_confidence(self,
                                         original: torch.Tensor,
                                         normalized: torch.Tensor,
                                         std_threshold: float = 2.0) -> float:
        """
        정상화 신뢰도 계산
        - 정상 범위를 벗어난 값들이 얼마나 정상 범위로 돌아왔는지 계산
        
        Args:
            original: 원본 데이터
            normalized: 정상화된 데이터
            std_threshold: 정상 범위 기준 (표준편차의 몇 배)
        """
        # 각 변수별 정상 범위 계산
        means = torch.mean(original, dim=0)
        stds = torch.std(original, dim=0)
        
        # 정상 범위를 벗어난 값들의 마스크
        upper_bound = means + std_threshold * stds
        lower_bound = means - std_threshold * stds
        
        abnormal_mask = (original > upper_bound) | (original < lower_bound)
        
        # 정상화 후 정상 범위 내로 들어온 값들의 비율
        normalized_normal = (normalized <= upper_bound) & (normalized >= lower_bound)
        recovery_ratio = torch.sum(normalized_normal[abnormal_mask]).float() / \
                        torch.sum(abnormal_mask).float()
        
        return recovery_ratio.item()
    
    def evaluate_fault_type(self,
                          m_seq: torch.Tensor,
                          fault_type: torch.Tensor,
                          fault_time: torch.Tensor) -> Dict[str, float]:
        """
        특정 Fault 타입에 대한 정상화 성능 평가
        
        Args:
            m_seq: 조작변수 시퀀스 (B, T, M)
            fault_type: Fault 타입 (B, fault_dim)
            fault_time: Fault 발생 시점 (B,)
        """
        self.model.eval()
        with torch.no_grad():
            # 모델 예측
            normalized_m = self.model(m_seq, fault_type, fault_time)
            
            # Fault 발생 구간 마스크 생성
            batch_size, seq_len = m_seq.size(0), m_seq.size(1)
            fault_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
            for i, t in enumerate(fault_time):
                fault_mask[i, t:] = True
            
            # 메트릭 계산
            metrics = self.calculate_metrics(m_seq, normalized_m, fault_mask)
            
        return metrics
    
    def evaluate_all_faults(self,
                          test_loader: torch.utils.data.DataLoader,
                          save_dir: str = None) -> Dict[str, Dict[str, float]]:
        """
        모든 Fault 타입에 대한 종합 평가
        
        Args:
            test_loader: 테스트 데이터 로더
            save_dir: 결과 저장 디렉토리 (옵션)
        """
        fault_metrics = {}
        
        for batch in test_loader:
            m_seq = batch['m_seq'].to(self.device)
            fault_type = batch['fault_type'].to(self.device)
            fault_time = batch['fault_time'].to(self.device)
            
            # Fault 타입 인덱스 추출
            fault_idx = torch.argmax(fault_type[0]).item()
            
            # 해당 Fault 타입 평가
            metrics = self.evaluate_fault_type(m_seq, fault_type, fault_time)
            
            if fault_idx not in fault_metrics:
                fault_metrics[fault_idx] = {
                    'samples': 0,
                    'metrics': {k: 0.0 for k in metrics.keys()}
                }
            
            # 메트릭 누적
            fault_metrics[fault_idx]['samples'] += 1
            for k, v in metrics.items():
                fault_metrics[fault_idx]['metrics'][k] += v
        
        # 평균 계산
        for fault_idx in fault_metrics:
            samples = fault_metrics[fault_idx]['samples']
            for k in fault_metrics[fault_idx]['metrics']:
                fault_metrics[fault_idx]['metrics'][k] /= samples
        
        # 결과 저장
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 메트릭 저장
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump(fault_metrics, f, indent=2)
            
            # 시각화
            self._plot_metrics(fault_metrics, save_dir)
        
        return fault_metrics
    
    def _plot_metrics(self, fault_metrics: Dict[str, Dict[str, float]], save_dir: str):
        """평가 결과 시각화"""
        # 1. MSE, MAE, RMSE 비교
        basic_metrics = ['mse', 'mae', 'rmse']
        fault_types = list(fault_metrics.keys())
        
        fig, axes = plt.subplots(1, len(basic_metrics), figsize=(15, 5))
        for i, metric in enumerate(basic_metrics):
            values = [fault_metrics[f]['metrics'][metric] for f in fault_types]
            axes[i].bar(fault_types, values)
            axes[i].set_title(f'{metric.upper()} by Fault Type')
            axes[i].set_xlabel('Fault Type')
            axes[i].set_ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'basic_metrics.png'))
        plt.close()
        
        # 2. 변수별 정규화 효과 히트맵
        var_metrics = [k for k in fault_metrics[0]['metrics'].keys() 
                      if k.startswith('mv') and k.endswith('reduction')]
        values = np.zeros((len(fault_types), len(var_metrics)))
        
        for i, fault in enumerate(fault_types):
            for j, metric in enumerate(var_metrics):
                values[i, j] = fault_metrics[fault]['metrics'][metric]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(values, 
                   xticklabels=[v.replace('_reduction', '') for v in var_metrics],
                   yticklabels=fault_types,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlBu')
        plt.title('Variance Reduction (%) by Variable and Fault Type')
        plt.xlabel('Manipulated Variables')
        plt.ylabel('Fault Type')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'variance_reduction.png'))
        plt.close()
        
        # 3. 정상화 신뢰도 점수
        plt.figure(figsize=(10, 6))
        confidence_scores = [fault_metrics[f]['metrics']['normalization_confidence'] 
                           for f in fault_types]
        plt.bar(fault_types, confidence_scores)
        plt.title('Normalization Confidence Score by Fault Type')
        plt.xlabel('Fault Type')
        plt.ylabel('Confidence Score')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_scores.png'))
        plt.close()


@click.command()
@click.option('--model_path', required=True, type=str, help='학습된 모델 경로 (.pth 파일)')
@click.option('--test_data', type=str, default='data/test_X_tr.npy', help='테스트 데이터 NPY 파일 경로')
@click.option('--test_labels', type=str, default='data/test_y_tr.npy', help='테스트 라벨 NPY 파일 경로')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=32, help='배치 크기')
@click.option('--save_dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
def main(model_path, test_data, test_labels, cuda, batch_size, save_dir):
    """
    Transformer 모델 평가 메인 함수
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger()
    
    # GPU 설정
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"장치: {device}")
    
    # 테스트 데이터셋 생성
    test_dataset = TEPNPYDataset(
        data_path=test_data,
        labels_path=test_labels,  # fault 유형별 시각화를 위해 라벨 필요
        transform=CSVToTensor(),
        is_test=True  # 50 윈도우 크기 사용
    )
    
    # 데이터셋 정보 출력
    logger.info("\n데이터셋 정보:")
    first_data = test_dataset[0]
    logger.info(f"입력 데이터 shape: {first_data['shot'].shape}")  # [50, 52]
    logger.info(f"측정값(m) shape: {first_data['shot'][:, :11].shape}")  # [50, 11]
    logger.info(f"라벨 shape: {first_data['label'].shape}")  # []
    
    # 데이터 로더 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 첫 번째 배치 shape 출력
    first_batch = next(iter(test_loader))
    logger.info("\n첫 번째 배치 정보:")
    logger.info(f"배치 전체 shape: {first_batch['shot'].shape}")  # [B, 50, 52]
    logger.info(f"배치 측정값(m) shape: {first_batch['shot'][:, :, :11].shape}")  # [B, 50, 11]
    logger.info(f"배치 라벨 shape: {first_batch['label'].shape}")  # [B]
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 평가
    logger.info("모델 평가 시작...")
    results = evaluate_reconstruction(model, test_loader, device, save_dir)
    
    # 결과 저장
    save_results(results, save_dir)
    logger.info(f"결과가 {save_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main() 