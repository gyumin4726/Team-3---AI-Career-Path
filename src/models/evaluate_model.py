#!/usr/bin/env python3
"""
Tennessee Eastman Process NPY 데이터 평가 스크립트
학습된 GAN v5 모델을 사용하여 NPY 테스트 데이터 평가
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import click
import logging
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from src.data.dataset import TEP_MEAN, TEP_STD, CSVToTensor, CSVNormalize, TEPNPYDataset, TEPDataset
from src.models.convolutional_models import CNN1D2DDiscriminatorMultitask


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
    
    # 체크포인트에서 discriminator state dict 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # Discriminator 모델 생성
    model = CNN1D2DDiscriminatorMultitask(
        input_size=52,  # TEP 데이터의 feature 수
        n_layers_1d=4,
        n_layers_2d=4,
        n_channel=52 * 3,
        n_channel_2d=100,
        class_count=13,  # 정상 + 12개 결함
        kernel_size=9,
        dropout=0.2,
        groups=52
    ).to(device)
    
    # state dict 로드
    model.load_state_dict(checkpoint['discriminator_state_dict'])
    model.eval()
    
    logger.info(f"모델 로드 완료: {type(model).__name__}")
    return model


def print_sequential_voting_results(predictions_by_sim, labels_by_sim, logger, save_dir):
    """시뮬레이션별 순차적 윈도우 투표 결과 출력 및 저장"""
    logger.info("\n시뮬레이션별 순차적 윈도우 투표 결과 (처음 5개 시뮬레이션):")
    
    # 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'sequential_voting_results.txt'), 'w', encoding='utf-8') as f:
        # 처음 5개 시뮬레이션에 대해 출력 및 저장
        for sim_idx in sorted(list(predictions_by_sim.keys()))[:5]:
            true_label = labels_by_sim[sim_idx]
            predictions = predictions_by_sim[sim_idx]
            
            header = f"\n시뮬레이션 {sim_idx} (실제 클래스: {true_label}):"
            logger.info(header)
            f.write(header + '\n')
            
            subheader = "윈도우별 예측:"
            logger.info(subheader)
            f.write(subheader + '\n')
            
            # 각 윈도우의 예측값을 순서대로 출력
            for window_idx, pred in enumerate(predictions):
                line = f"  윈도우 {window_idx:3d}: 예측 = {pred}"
                logger.info(line)
                f.write(line + '\n')
            
            # 전체 윈도우에 대한 최종 투표 결과
            final_votes = Counter(predictions)
            final_pred = final_votes.most_common(1)[0][0]
            
            summary = f"\n  최종 결과:"
            logger.info(summary)
            f.write(summary + '\n')
            
            vote_dist = f"    전체 투표 분포: {dict(final_votes)}"
            logger.info(vote_dist)
            f.write(vote_dist + '\n')
            
            final = f"    최종 예측: {final_pred}"
            logger.info(final)
            f.write(final + '\n')
            
            accuracy = f"    예측 정확성: {'정확' if final_pred == true_label else '오류'}"
            logger.info(accuracy)
            f.write(accuracy + '\n')


def evaluate_model(model, test_loader, device):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        device: 연산 장치 (CPU/GPU)
        
    Returns:
        accuracy: 전체 시뮬레이션에 대한 정확도
        predictions: 시뮬레이션별 예측 결과
        true_labels: 시뮬레이션별 실제 라벨
    """
    model.eval()
    predictions_by_sim = {}  # 시뮬레이션별 예측값 저장
    labels_by_sim = {}      # 시뮬레이션별 실제 라벨 저장
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["shot"].to(device)
            labels = batch["label"].to(device)
            sim_indices = batch["sim_idx"].to(device)
            
            # 모델 예측
            type_logits, _ = model(inputs, sim_indices)
            type_logits = type_logits.transpose(1, 2)
            batch_preds = torch.argmax(type_logits, dim=1)
            
            # CPU로 이동하고 numpy로 변환
            batch_preds = batch_preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            sim_indices = sim_indices.cpu().numpy()
            
            # 시뮬레이션별로 예측값 수집 (순서 유지)
            for i, (pred, label, idx) in enumerate(zip(batch_preds, batch_labels, sim_indices)):
                if idx not in predictions_by_sim:
                    predictions_by_sim[idx] = []
                    labels_by_sim[idx] = label
                predictions_by_sim[idx].extend(pred)
    
    # 순차적 투표 결과 출력
    logger = logging.getLogger(__name__)
    print_sequential_voting_results(predictions_by_sim, labels_by_sim, logger, save_dir)
    
    # 각 시뮬레이션별로 majority voting 수행
    final_predictions = []
    final_labels = []
    
    for sim_idx in sorted(predictions_by_sim.keys()):
        final_pred = Counter(predictions_by_sim[sim_idx]).most_common(1)[0][0]
        final_predictions.append(final_pred)
        final_labels.append(labels_by_sim[sim_idx])
    
    # numpy 배열로 변환
    final_predictions = np.array(final_predictions)
    final_labels = np.array(final_labels)
    
    # 정확도 계산
    accuracy = (final_predictions == final_labels).mean()
    
    return accuracy, final_predictions, final_labels


def analyze_results(predictions, labels, logger):
    """Run-level result analysis and metric calculation"""
    logger.info("런 단위 결과 분석 시작...")
    
    # 정확도 계산
    accuracy = accuracy_score(labels, predictions)
    logger.info(f"전체 정확도: {accuracy:.4f}")
    
    # 클래스별 성능 지표 계산
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions)
    
    # 결과 출력
    logger.info("\n클래스별 성능:")
    for i in range(len(precision)):
        if i == 0:
            logger.info(f"클래스 {i} (정상):")
        else:
            logger.info(f"클래스 {i} (결함{i}):")
        logger.info(f"  - 정밀도: {precision[i]:.4f}")
        logger.info(f"  - 재현율: {recall[i]:.4f}")
        logger.info(f"  - F1 점수: {f1[i]:.4f}")
        logger.info(f"  - 샘플 수: {support[i]}")
    
    # 분류 보고서
    logger.info("\n상세 분류 보고서:")
    logger.info("\n" + classification_report(labels, predictions))
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }


def plot_confusion_matrix(run_predictions, run_labels, save_path=None):
    """
    혼동 행렬 시각화
    
    Args:
        run_predictions: 예측 라벨
        run_labels: 실제 라벨
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(run_labels, run_predictions)
    
    # 정확도 계산
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\n정확도: {accuracy:.4f}")
    
    # 클래스 라벨 생성
    classes = [f"FaultFree" if i == 0 else f"Fault{i}" for i in range(len(np.unique(run_labels)))]
    
    # 히트맵 생성
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        # 저장 경로의 디렉토리가 없으면 생성
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_results(results, save_dir):
    """
    평가 결과 저장
    
    Args:
        results: 평가 결과 딕셔너리
        save_dir: 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 결과를 JSON 형식으로 저장
    results_path = os.path.join(save_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)


@click.command()
@click.option('--model_path', type=str, default='model_pretrained/model1/19_epoch_checkpoint.pth', help='학습된 discriminator 모델 경로 (.pth 파일)')
@click.option('--test_data', type=str, default='data/test_X_model1.npy', help='테스트 데이터 NPY 파일 경로')
@click.option('--test_labels', type=str, default='data/test_y_model1.npy', help='테스트 라벨 NPY 파일 경로')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=16, help='배치 크기')
@click.option('--save_dir', type=str, default='model1_evaluation_results', help='결과 저장 디렉토리')
@click.option('--random_seed', type=int, default=42, help='랜덤 시드')
def main(model_path, test_data, test_labels, cuda, batch_size, save_dir, random_seed):
    """
    모델 평가 메인 함수
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 랜덤 시드 설정
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 로거 설정
    logger = setup_logger()
    
    # GPU 설정
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"장치: {device}")
    
    # 데이터 변환 설정
    transform = transforms.Compose([
        CSVToTensor()
    ])
    
    # 테스트 데이터셋 생성
    test_dataset = TEPNPYDataset(
        data_path=test_data,
        labels_path=test_labels,
        transform=transform,
        is_test=True  # 테스트 데이터는 92개 윈도우 사용
    )
    
    # 데이터 로더 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Windows에서의 multiprocessing 문제 해결을 위해 0으로 설정
    )
    
    # 데이터셋 상세 정보 확인
    logger.info("\n데이터셋 상세 정보:")
    
    # 첫 번째 배치 가져오기
    first_batch = next(iter(test_loader))
    logger.info(f"배치 데이터 shape: {first_batch['shot'].shape}")  # [batch_size, 50, 52]
    logger.info(f"배치 라벨 shape: {first_batch['label'].shape}")   # [batch_size, 13]
    logger.info(f"배치 시뮬레이션 인덱스 shape: {first_batch['sim_idx'].shape}")  # [batch_size]
    
    # 시뮬레이션 인덱스 분포 확인
    sim_indices = first_batch['sim_idx'].numpy()
    unique_sims = np.unique(sim_indices)
    logger.info(f"\n시뮬레이션 인덱스 정보:")
    logger.info(f"배치 내 고유 시뮬레이션 수: {len(unique_sims)}")
    logger.info(f"시뮬레이션 인덱스 범위: {sim_indices.min()} ~ {sim_indices.max()}")
    
    # 첫 번째 배치의 일부 데이터 출력
    logger.info("\n첫 번째 배치 샘플 (처음 5개):")
    for i in range(min(5, len(sim_indices))):
        logger.info(f"샘플 {i}:")
        logger.info(f"  - 시뮬레이션 인덱스: {sim_indices[i]}")
        logger.info(f"  - 라벨: {first_batch['label'][i].argmax().item()}")  # 원-핫 인코딩된 라벨의 클래스
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 평가
    logger.info("모델 평가 시작...")
    accuracy, predictions, labels = evaluate_model(model, test_loader, device)
    logger.info(f"평가 완료. 정확도: {accuracy:.4f}")
    
    # 결과 분석
    results = analyze_results(predictions, labels, logger)
    
    # 혼동 행렬 플롯 저장
    plot_confusion_matrix(
        predictions,
        labels,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # 결과 저장
    save_results(results, save_dir)
    logger.info(f"결과가 {save_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main() 