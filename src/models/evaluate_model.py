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
    
    # 전체 모델 로드
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    logger.info(f"모델 로드 완료: {type(model).__name__}")
    return model


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
            sim_indices = batch["sim_idx"].to(device)  # 학습 시와 동일하게 처리
            
            # 모델 예측 (train과 동일하게 sim_indices 사용)
            type_logits, _ = model(inputs, sim_indices)
            type_logits = type_logits.transpose(1, 2)  # train과 동일한 형식으로 변경
            batch_preds = torch.argmax(type_logits, dim=1)  # (batch_size, n_classes, seq_len) -> (batch_size, seq_len)
            
            # CPU로 이동하고 numpy로 변환
            batch_preds = batch_preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            # 시뮬레이션별로 예측값 수집
            sim_idx = batch["sim_idx"].cpu().numpy() if "sim_idx" in batch else np.arange(len(batch_preds))
            for i, (pred, label, idx) in enumerate(zip(batch_preds, batch_labels, sim_idx)):
                if idx not in predictions_by_sim:
                    predictions_by_sim[idx] = []
                    labels_by_sim[idx] = label[0]  # 각 시뮬레이션의 라벨은 동일
                predictions_by_sim[idx].extend(pred)
    
    # 각 시뮬레이션별로 majority voting 수행
    final_predictions = []
    final_labels = []
    
    for sim_idx in sorted(predictions_by_sim.keys()):
        # Majority voting으로 최종 예측
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
    
    # 클래스 라벨 생성
    classes = [f"정상" if i == 0 else f"결함{i}" for i in range(len(np.unique(run_labels)))]
    
    # 히트맵 생성
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('혼동 행렬')
    plt.xlabel('예측')
    plt.ylabel('실제')
    
    if save_path:
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
@click.option('--model_path', required=True, type=str, help='학습된 discriminator 모델 경로 (.pth 파일)')
@click.option('--test_data', type=str, default='data/test_data.npy', help='테스트 데이터 NPY 파일 경로')
@click.option('--test_labels', type=str, default='data/test_labels.npy', help='테스트 라벨 NPY 파일 경로')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=16, help='배치 크기')
@click.option('--save_dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
@click.option('--random_seed', type=int, default=42, help='랜덤 시드')
def main(model_path, test_data, test_labels, cuda, batch_size, save_dir, random_seed):
    """
    모델 평가 메인 함수
    """
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
        transform=transform
    )
    
    # 데이터 로더 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
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