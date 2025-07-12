#!/usr/bin/env python3
"""
Tennessee Eastman Process 정상화 C-TCN-AE 모델 평가 스크립트
고장 데이터를 정상으로 재구성하는 성능 평가
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import click
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from mpl_toolkits.mplot3d import Axes3D

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
    """학습된 정상화 C-TCN-AE 모델 로드"""
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
    
    logger.info(f"정상화 모델 로드 완료: {type(model).__name__}")
    return model


def evaluate_normalization_performance(model, test_loader, normal_dataset, device, save_dir):
    """정상화 모델의 성능 평가 - 고장 데이터를 정상으로 재구성"""
    logger = logging.getLogger(__name__)
    model.eval()
    
    # 결과 저장용
    fault_results = defaultdict(lambda: {
        'original_seqs': [],
        'normalized_seqs': [],
        'normal_targets': [],
        'mse_scores': [],
        'mae_scores': [],
        'samples': 0
    })
    
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 고장 데이터 (입력)
            m_seq = batch["shot"][:, :, :11].to(device)  # [B, 50, 11] - 고장 데이터
            fault_labels = batch["label"].to(device)  # [B,] - 고장 라벨 (1~12)
            batch_size = m_seq.size(0)  # 현재 배치 크기
            
            # 고장 데이터의 인덱스를 기반으로 정상 데이터 가져오기 (학습과 동일한 방식)
            fault_indices = batch.get("index", torch.arange(len(batch["shot"])))  # 배치 내 인덱스
            
            # 정상 데이터 가져오기 (고장과 같은 순서)
            normal_seqs = []
            for idx in fault_indices:
                # 고장 인덱스를 정상 인덱스로 변환 (같은 시뮬레이션)
                normal_idx = idx % len(normal_dataset)  # 정상 데이터셋 크기로 모듈로
                normal_data = normal_dataset[normal_idx]
                normal_seqs.append(normal_data["shot"][:, :11])  # 측정값만 사용
            
            normal_seq = torch.stack(normal_seqs).to(device)  # [B, 50, 11] - 정상 데이터
            
            # 첫 배치의 shape 출력
            if i == 0:
                logger.info("\n데이터 shape 정보:")
                logger.info(f"고장 입력 shape: {m_seq.shape}")
                logger.info(f"고장 라벨 shape: {fault_labels.shape}")
                logger.info(f"고장 라벨 고유값: {torch.unique(fault_labels).cpu().numpy()}")
                logger.info(f"정상 타겟 shape: {normal_seq.shape}")
            
            # 정상화 모델: 고장 데이터를 정상으로 재구성
            normal_condition = torch.zeros_like(fault_labels)  # 모든 조건을 정상(0)으로
            m_normalized = model(m_seq, normal_condition)
            
            if i == 0:
                logger.info(f"정상화된 데이터 shape: {m_normalized.shape}")
            
            # CPU로 이동하고 numpy로 변환
            m_seq_cpu = m_seq.cpu().numpy()
            m_normalized_cpu = m_normalized.cpu().numpy()
            normal_seq_cpu = normal_seq.cpu().numpy()
            fault_labels_cpu = fault_labels.cpu().numpy()
            
            # 각 고장 타입별로 결과 저장
            for j, fault_label in enumerate(fault_labels_cpu):
                fault_type = int(fault_label)
                if fault_type > 0:  # 고장 데이터만 평가 (1~12)
                    fault_results[fault_type]['original_seqs'].append(m_seq_cpu[j])
                    fault_results[fault_type]['normalized_seqs'].append(m_normalized_cpu[j])
                    fault_results[fault_type]['normal_targets'].append(normal_seq_cpu[j])
                    
                    # MSE, MAE 계산 - 고장 데이터를 정상으로 재구성하는 성능
                    mse = mean_squared_error(normal_seq_cpu[j].flatten(), m_normalized_cpu[j].flatten())
                    mae = mean_absolute_error(normal_seq_cpu[j].flatten(), m_normalized_cpu[j].flatten())
                    
                    fault_results[fault_type]['mse_scores'].append(mse)
                    fault_results[fault_type]['mae_scores'].append(mae)
                    fault_results[fault_type]['samples'] += 1
                    total_samples += 1
    
    # 결과 정리
    logger.info(f"\n총 평가 샘플 수: {total_samples}")
    
    # 각 고장 타입별 통계
    fault_statistics = {}
    logger.info("\n고장 타입별 정상화 성능:")
    
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
    """정상화 결과 시각화"""
    logger = logging.getLogger(__name__)
    logger.info("\n시각화 생성 중...")
    
    # 1. 고장 타입별 MSE, MAE 비교
    fault_types = sorted(fault_statistics.keys())
    mse_values = [fault_statistics[ft]['avg_mse'] for ft in fault_types]
    mae_values = [fault_statistics[ft]['avg_mae'] for ft in fault_types]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE 비교
    bars1 = ax1.bar(fault_types, mse_values, color='lightgreen', alpha=0.7)
    ax1.set_title('고장 타입별 정상화 MSE (고장→정상)')
    ax1.set_xlabel('고장 타입')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # MAE 비교
    bars2 = ax2.bar(fault_types, mae_values, color='lightblue', alpha=0.7)
    ax2.set_title('고장 타입별 정상화 MAE (고장→정상)')
    ax2.set_xlabel('고장 타입')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fault_normalization_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 3차원 t-SNE 시각화 (모든 고장 타입 통합)
    plot_3d_tsne_visualization(fault_results, save_dir)
    
    # 3. 변수별 정상화 효과 히트맵
    plot_variable_normalization_heatmap(fault_results, fault_statistics, save_dir)
    
    logger.info("시각화 완료!")


def plot_3d_tsne_visualization(fault_results, save_dir):
    """3차원 t-SNE 시각화 - 모든 고장 타입의 원본 vs 정상화 분포"""
    logger = logging.getLogger(__name__)
    logger.info("3차원 t-SNE 시각화 생성 중...")
    
    # 모든 데이터 수집
    all_original = []
    all_normalized = []
    all_labels = []
    
    for fault_type in sorted(fault_results.keys()):
        if len(fault_results[fault_type]['original_seqs']) > 0:
            original_seqs = np.array(fault_results[fault_type]['original_seqs'])  # [N, 50, 11]
            normalized_seqs = np.array(fault_results[fault_type]['normalized_seqs'])  # [N, 50, 11]
            
            # Flatten sequences
            orig_flat = original_seqs.reshape(original_seqs.shape[0], -1)  # [N, 50*11]
            norm_flat = normalized_seqs.reshape(normalized_seqs.shape[0], -1)  # [N, 50*11]
            
            all_original.append(orig_flat)
            all_normalized.append(norm_flat)
            all_labels.extend([fault_type] * len(orig_flat))
    
    # 데이터 결합
    all_original = np.vstack(all_original)  # [Total_N, 550]
    all_normalized = np.vstack(all_normalized)  # [Total_N, 550]
    all_labels = np.array(all_labels)
    
    logger.info(f"t-SNE 시각화 데이터 크기: {all_original.shape}")
    logger.info(f"고장 타입: {np.unique(all_labels)}")
    
    # t-SNE 적용 (원본 데이터로 학습)
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_original = tsne.fit_transform(all_original)
    tsne_normalized = tsne.transform(all_normalized)
    
    # 3차원 시각화
    fig = plt.figure(figsize=(15, 10))
    
    # 원본 데이터 시각화
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(all_labels))))
    
    for i, fault_type in enumerate(sorted(np.unique(all_labels))):
        mask = all_labels == fault_type
        ax1.scatter(tsne_original[mask, 0], tsne_original[mask, 1], tsne_original[mask, 2],
                   label=f'Fault {fault_type} (Original)', 
                   color=colors[i], alpha=0.7, s=30)
    
    ax1.set_title('Original Fault Distributions (3D t-SNE)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_zlabel('t-SNE 3')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 정상화된 데이터 시각화
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, fault_type in enumerate(sorted(np.unique(all_labels))):
        mask = all_labels == fault_type
        ax2.scatter(tsne_normalized[mask, 0], tsne_normalized[mask, 1], tsne_normalized[mask, 2],
                   label=f'Fault {fault_type} (Normalized)', 
                   color=colors[i], alpha=0.7, s=30, marker='^')
    
    ax2.set_title('Normalized Fault Distributions (3D t-SNE)')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_zlabel('t-SNE 3')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fault_normalization_3d_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 추가: 원본 vs 정상화 비교 시각화
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, fault_type in enumerate(sorted(np.unique(all_labels))):
        mask = all_labels == fault_type
        # 원본 데이터
        ax.scatter(tsne_original[mask, 0], tsne_original[mask, 1], tsne_original[mask, 2],
                  label=f'Fault {fault_type} (Original)', 
                  color=colors[i], alpha=0.6, s=30)
        # 정상화된 데이터
        ax.scatter(tsne_normalized[mask, 0], tsne_normalized[mask, 1], tsne_normalized[mask, 2],
                  label=f'Fault {fault_type} (Normalized)', 
                  color=colors[i], alpha=0.6, s=30, marker='^')
    
    ax.set_title('Fault Normalization Comparison (3D t-SNE)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fault_normalization_comparison_3d_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_variable_normalization_heatmap(fault_results, fault_statistics, save_dir):
    """변수별 정상화 효과 히트맵"""
    fault_types = sorted(fault_statistics.keys())
    n_variables = 11
    
    # 변수별 MSE 계산
    variable_mse = np.zeros((len(fault_types), n_variables))
    
    for i, fault_type in enumerate(fault_types):
        if len(fault_results[fault_type]['normalized_seqs']) > 0:
            normalized_seqs = np.array(fault_results[fault_type]['normalized_seqs'])  # [N, 50, 11]
            normal_targets = np.array(fault_results[fault_type]['normal_targets'])  # [N, 50, 11]
            
            # 각 변수별 MSE 계산
            for j in range(n_variables):
                norm_var = normalized_seqs[:, :, j].flatten()  # [N*50]
                target_var = normal_targets[:, :, j].flatten()  # [N*50]
                variable_mse[i, j] = mean_squared_error(target_var, norm_var)
    
    # 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(variable_mse, 
               xticklabels=[f'MV{i+1}' for i in range(n_variables)],
               yticklabels=[f'Fault {ft}' for ft in fault_types],
               annot=True, 
               fmt='.4f',
               cmap='RdYlBu_r',
               cbar_kws={'label': 'MSE'})
    plt.title('Variable-wise Normalization MSE (Fault → Normal)')
    plt.xlabel('Manipulated Variables')
    plt.ylabel('Fault Types')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'variable_normalization_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, save_dir):
    """평가 결과 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 결과를 JSON 형식으로 저장
    results_path = os.path.join(save_dir, 'normalization_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 요약 통계 저장
    summary_stats = {
        'total_fault_types': len(results),
        'avg_mse_across_faults': np.mean([results[ft]['avg_mse'] for ft in results]),
        'avg_mae_across_faults': np.mean([results[ft]['avg_mae'] for ft in results]),
        'best_normalization_fault': min(results.keys(), key=lambda x: results[x]['avg_mse']),
        'worst_normalization_fault': max(results.keys(), key=lambda x: results[x]['avg_mse'])
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
    정상화 C-TCN-AE 모델 평가 메인 함수
    고장 데이터를 정상으로 재구성하는 성능 평가
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger()
    
    # GPU 설정
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"장치: {device}")
    
    # 전체 테스트 데이터셋 생성
    full_test_dataset = TEPNPYDataset(
        data_path=test_data,
        labels_path=test_labels,
        transform=CSVToTensor(),
        is_test=True
    )
    
    # 정상 데이터만 필터링
    normal_indices = [i for i, label in enumerate(full_test_dataset.labels) if label == 0]
    normal_dataset = torch.utils.data.Subset(full_test_dataset, normal_indices)
    
    # 고장 데이터만 필터링 (정상 제외)
    fault_indices = [i for i, label in enumerate(full_test_dataset.labels) if label > 0]
    fault_dataset = torch.utils.data.Subset(full_test_dataset, fault_indices)
    
    # 데이터셋 정보 출력
    logger.info("\n데이터셋 정보:")
    logger.info(f"전체 테스트 데이터셋 크기: {len(full_test_dataset)}")
    logger.info(f"정상 테스트 데이터셋 크기: {len(normal_dataset)}")
    logger.info(f"고장 테스트 데이터셋 크기: {len(fault_dataset)}")
    
    # 데이터 로더 생성
    fault_loader = DataLoader(
        fault_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 첫 번째 배치 정보 출력
    first_fault_batch = next(iter(fault_loader))
    first_normal_data = normal_dataset[0]  # 첫 번째 정상 데이터
    
    logger.info("\n첫 번째 배치 정보:")
    logger.info(f"고장 배치 shape: {first_fault_batch['shot'].shape}")
    logger.info(f"고장 라벨 shape: {first_fault_batch['label'].shape}")
    logger.info(f"고장 라벨 고유값: {torch.unique(first_fault_batch['label']).numpy()}")
    logger.info(f"정상 데이터 shape: {first_normal_data['shot'].shape}")
    logger.info(f"정상 라벨: {first_normal_data['label']}")
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 평가 (정상화 성능)
    logger.info("정상화 성능 평가 시작...")
    results = evaluate_normalization_performance(model, fault_loader, normal_dataset, device, save_dir)
    
    # 결과 저장
    save_results(results, save_dir)
    logger.info(f"결과가 {save_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main() 