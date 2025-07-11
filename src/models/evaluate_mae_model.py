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