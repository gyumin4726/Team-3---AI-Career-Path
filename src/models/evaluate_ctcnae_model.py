#!/usr/bin/env python3
"""
Tennessee Eastman Process C-TCN-AE 모델 평가 스크립트
Conditional Temporal Convolutional Network Autoencoder를 사용하여 고장 데이터를 정상으로 재구성
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
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
from src.models.ctcnae_models import ConditionalTCNAutoencoder


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model(model_path, device):
    """학습된 C-TCN-AE 모델 로드"""
    logger = logging.getLogger(__name__)
    logger.info(f"모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 체크포인트에서 모델 state dict 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 생성
    model = ConditionalTCNAutoencoder(
        m_dim=11,  # 측정값만 사용
        fault_dim=13,  # fault 타입 수
        channels=[32, 64, 64],
        kernel_size=3
    ).to(device)
    
    # state dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"모델 로드 완료: {type(model).__name__}")
    return model


def evaluate_fault_to_normal_reconstruction(model, test_loader, device, save_dir):
    """고장 데이터를 정상으로 재구성하는 성능 평가"""
    logger = logging.getLogger(__name__)
    model.eval()
    
    # 결과 저장용
    fault_results = defaultdict(lambda: {
        'original_seqs': [],
        'reconstructed_seqs': [],
        'mse_scores': [],
        'mae_scores': [],
        'samples': 0
    })
    
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 측정값(m)과 라벨 사용
            m_seq = batch["shot"][:, :, :11].to(device)  # [B, 50, 11] - 측정값만 사용
            fault_labels = batch["label"].to(device)  # [B,] - 고장 라벨 (1~12)
            
            # 첫 배치의 shape 출력
            if i == 0:
                logger.info("\n데이터 shape 정보:")
                logger.info(f"입력 데이터 shape: {m_seq.shape}")  # [B, 50, 11]
                logger.info(f"고장 라벨 shape: {fault_labels.shape}")  # [B]
                logger.info(f"고장 라벨 고유값: {torch.unique(fault_labels).cpu().numpy()}")
            
            # 정상 라벨로 재구성 (모든 고장을 0으로 변환)
            normal_labels = torch.zeros_like(fault_labels)  # [B,] - 모두 0 (정상)
            
            # 재구성 [B, 50, 11]
            m_rec = model(m_seq, normal_labels)
            
            if i == 0:
                logger.info(f"재구성 데이터 shape: {m_rec.shape}")  # [B, 50, 11]
            
            # CPU로 이동하고 numpy로 변환
            m_seq_cpu = m_seq.cpu().numpy()
            m_rec_cpu = m_rec.cpu().numpy()
            fault_labels_cpu = fault_labels.cpu().numpy()
            
            # 각 고장 타입별로 결과 저장
            for j, fault_label in enumerate(fault_labels_cpu):
                fault_type = int(fault_label)
                if fault_type > 0:  # 고장 데이터만 평가 (1~12)
                    fault_results[fault_type]['original_seqs'].append(m_seq_cpu[j])
                    fault_results[fault_type]['reconstructed_seqs'].append(m_rec_cpu[j])
                    
                    # MSE, MAE 계산
                    mse = mean_squared_error(m_seq_cpu[j].flatten(), m_rec_cpu[j].flatten())
                    mae = mean_absolute_error(m_seq_cpu[j].flatten(), m_rec_cpu[j].flatten())
                    
                    fault_results[fault_type]['mse_scores'].append(mse)
                    fault_results[fault_type]['mae_scores'].append(mae)
                    fault_results[fault_type]['samples'] += 1
                    total_samples += 1
    
    # 결과 정리
    logger.info(f"\n총 평가 샘플 수: {total_samples}")
    
    # 각 고장 타입별 통계
    fault_statistics = {}
    logger.info("\n고장 타입별 재구성 성능:")
    
    for fault_type in sorted(fault_results.keys()):
        if fault_results[fault_type]['samples'] > 0:
            avg_mse = np.mean(fault_results[fault_type]['mse_scores'])
            avg_mae = np.mean(fault_results[fault_type]['mae_scores'])
            std_mse = np.std(fault_results[fault_type]['mse_scores'])
            std_mae = np.std(fault_results[fault_type]['mae_scores'])
            
            fault_statistics[fault_type] = {
                'avg_mse': float(avg_mse),
                'avg_mae': float(avg_mae),
                'std_mse': float(std_mse),
                'std_mae': float(std_mae),
                'samples': fault_results[fault_type]['samples']
            }
            
            logger.info(f"Fault {fault_type}: MSE={avg_mse:.6f}±{std_mse:.6f}, "
                       f"MAE={avg_mae:.6f}±{std_mae:.6f} "
                       f"(샘플 수: {fault_results[fault_type]['samples']})")
    
    # 시각화
    create_visualizations(fault_results, fault_statistics, save_dir)
    
    return fault_statistics


def create_visualizations(fault_results, fault_statistics, save_dir):
    """재구성 결과 시각화"""
    logger = logging.getLogger(__name__)
    logger.info("\n시각화 생성 중...")
    
    # 1. 고장 타입별 MSE, MAE 비교
    fault_types = sorted(fault_statistics.keys())
    mse_values = [fault_statistics[ft]['avg_mse'] for ft in fault_types]
    mae_values = [fault_statistics[ft]['avg_mae'] for ft in fault_types]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE 비교
    bars1 = ax1.bar(fault_types, mse_values, color='skyblue', alpha=0.7)
    ax1.set_title('고장 타입별 MSE (고장→정상 재구성)')
    ax1.set_xlabel('고장 타입')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # MAE 비교
    bars2 = ax2.bar(fault_types, mae_values, color='lightcoral', alpha=0.7)
    ax2.set_title('고장 타입별 MAE (고장→정상 재구성)')
    ax2.set_xlabel('고장 타입')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fault_reconstruction_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA 시각화 (각 고장 타입별)
    for fault_type in fault_types:
        if len(fault_results[fault_type]['original_seqs']) > 0:
            plot_fault_pca(fault_results[fault_type], fault_type, save_dir)
    
    # 3. 변수별 재구성 효과 히트맵
    plot_variable_reconstruction_heatmap(fault_results, fault_statistics, save_dir)
    
    logger.info("시각화 완료!")


def plot_fault_pca(fault_data, fault_type, save_dir):
    """특정 고장 타입의 PCA 시각화"""
    original_seqs = np.array(fault_data['original_seqs'])  # [N, 50, 11]
    reconstructed_seqs = np.array(fault_data['reconstructed_seqs'])  # [N, 50, 11]
    
    # Flatten sequences
    orig_flat = original_seqs.reshape(original_seqs.shape[0], -1)  # [N, 50*11]
    rec_flat = reconstructed_seqs.reshape(reconstructed_seqs.shape[0], -1)  # [N, 50*11]
    
    # PCA 학습 및 투영
    pca = PCA(n_components=2)
    pca.fit(orig_flat)
    
    orig_pca = pca.transform(orig_flat)  # [N, 2]
    rec_pca = pca.transform(rec_flat)    # [N, 2]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # 원본 고장 데이터
    plt.scatter(orig_pca[:, 0], orig_pca[:, 1], 
               label=f'Fault {fault_type} (Original)', 
               color='red', alpha=0.6, s=50)
    
    # 재구성된 정상 데이터
    plt.scatter(rec_pca[:, 0], rec_pca[:, 1], 
               label=f'Fault {fault_type} (Reconstructed as Normal)', 
               color='blue', alpha=0.6, s=50, marker='^')
    
    plt.title(f'Fault {fault_type} → Normal Reconstruction (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fault_{fault_type}_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_variable_reconstruction_heatmap(fault_results, fault_statistics, save_dir):
    """변수별 재구성 효과 히트맵"""
    fault_types = sorted(fault_statistics.keys())
    n_variables = 11
    
    # 변수별 MSE 계산
    variable_mse = np.zeros((len(fault_types), n_variables))
    
    for i, fault_type in enumerate(fault_types):
        if len(fault_results[fault_type]['original_seqs']) > 0:
            original_seqs = np.array(fault_results[fault_type]['original_seqs'])  # [N, 50, 11]
            reconstructed_seqs = np.array(fault_results[fault_type]['reconstructed_seqs'])  # [N, 50, 11]
            
            # 각 변수별 MSE 계산
            for j in range(n_variables):
                orig_var = original_seqs[:, :, j].flatten()  # [N*50]
                rec_var = reconstructed_seqs[:, :, j].flatten()  # [N*50]
                variable_mse[i, j] = mean_squared_error(orig_var, rec_var)
    
    # 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(variable_mse, 
               xticklabels=[f'MV{i+1}' for i in range(n_variables)],
               yticklabels=[f'Fault {ft}' for ft in fault_types],
               annot=True, 
               fmt='.4f',
               cmap='RdYlBu_r',
               cbar_kws={'label': 'MSE'})
    plt.title('Variable-wise Reconstruction MSE (Fault → Normal)')
    plt.xlabel('Manipulated Variables')
    plt.ylabel('Fault Types')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'variable_reconstruction_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, save_dir):
    """평가 결과 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 결과를 JSON 형식으로 저장
    results_path = os.path.join(save_dir, 'fault_to_normal_reconstruction_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 요약 통계 저장
    summary_stats = {
        'total_fault_types': len(results),
        'avg_mse_across_faults': np.mean([results[ft]['avg_mse'] for ft in results]),
        'avg_mae_across_faults': np.mean([results[ft]['avg_mae'] for ft in results]),
        'best_reconstruction_fault': min(results.keys(), key=lambda x: results[x]['avg_mse']),
        'worst_reconstruction_fault': max(results.keys(), key=lambda x: results[x]['avg_mse'])
    }
    
    summary_path = os.path.join(save_dir, 'summary_statistics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)


@click.command()
@click.option('--model_path', type=str, default='model_pretrained/model2/best_model.pth', help='학습된 C-TCN-AE 모델 경로 (.pth 파일)')
@click.option('--train_data', type=str, default='data/test_X_model2.npy', help='훈련 데이터 NPY 파일 경로')
@click.option('--train_labels', type=str, default='data/test_Y_model2.npy', help='훈련 라벨 NPY 파일 경로')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=128, help='배치 크기')
@click.option('--save_dir', type=str, default='evaluation_results/model2', help='결과 저장 디렉토리')
def main(model_path, test_data, test_labels, cuda, batch_size, save_dir):
    """
    C-TCN-AE 모델 평가 메인 함수
    고장 데이터(1~12)를 정상 데이터(0)로 재구성하는 성능 평가
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
        labels_path=test_labels,
        transform=CSVToTensor(),
        is_test=True  # 50 윈도우 크기 사용
    )
    
    # 데이터셋 정보 출력
    logger.info("\n데이터셋 정보:")
    first_data = test_dataset[0]
    logger.info(f"입력 데이터 shape: {first_data['shot'].shape}")  # [50, 52]
    logger.info(f"측정값(m) shape: {first_data['shot'][:, :11].shape}")  # [50, 11]
    logger.info(f"라벨: {first_data['label']}")
    
    # 전체 데이터셋 크기 출력
    logger.info(f"\n전체 테스트 데이터셋 크기:")
    logger.info(f"데이터셋 길이: {len(test_dataset)}")
    logger.info(f"전체 데이터 shape: {test_dataset.data.shape if hasattr(test_dataset, 'data') else 'N/A'}")
    logger.info(f"전체 라벨 shape: {test_dataset.labels.shape if hasattr(test_dataset, 'labels') else 'N/A'}")
    
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
    logger.info(f"라벨 고유값: {torch.unique(first_batch['label']).numpy()}")
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 평가 (고장→정상 재구성)
    logger.info("고장→정상 재구성 평가 시작...")
    results = evaluate_fault_to_normal_reconstruction(model, test_loader, device, save_dir)
    
    # 결과 저장
    save_results(results, save_dir)
    logger.info(f"결과가 {save_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main() 