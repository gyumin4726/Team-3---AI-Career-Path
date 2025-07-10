#!/usr/bin/env python3
"""
Tennessee Eastman Process CSV 데이터 평가 스크립트
학습된 GAN v5 모델을 사용하여 CSV 테스트 데이터 평가
960 시점 테스트 데이터를 평균 풀링으로 500 시점으로 압축하여 평가
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

from src.data.dataset import TEP_MEAN, TEP_STD, CSVToTensor, CSVNormalize, TEPCSVDataset


def average_pooling_1d(data, target_length):
    """
    1D 평균 풀링으로 시계열 데이터 압축
    Args:
        data: 원본 데이터 (timesteps, features)
        target_length: 목표 길이
    Returns:
        compressed_data: 압축된 데이터 (target_length, features)
    """
    original_length = data.shape[0]
    features = data.shape[1]
    
    # 구간 경계 계산 (균등 분할)
    boundaries = np.linspace(0, original_length, target_length + 1)
    
    compressed_data = np.zeros((target_length, features))
    
    for i in range(target_length):
        start_idx = int(boundaries[i])
        end_idx = int(boundaries[i + 1])
        
        # 각 구간의 평균 계산
        if start_idx == end_idx:
            # 구간이 너무 좁은 경우 (거의 발생하지 않음)
            compressed_data[i] = data[start_idx]
        else:
            compressed_data[i] = np.mean(data[start_idx:end_idx], axis=0)
    
    return compressed_data


class TEPCSVDatasetCompressed(Dataset):
    """
    CSV 파일로부터 TEP 데이터를 로드하고 평균 풀링으로 압축하는 데이터셋
    """
    def __init__(self, csv_files, transform=None, is_test=False):
        """
        Args:
            csv_files: CSV 파일 경로 리스트
            transform: 데이터 변환 함수
            is_test: 테스트 모드 여부 (960→500 압축 적용)
        """
        self.transform = transform
        self.is_test = is_test
        self.data = []
        self.labels = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # 시뮬레이션 런별로 그룹화
            for sim_run in df['simulationRun'].unique():
                run_data = df[df['simulationRun'] == sim_run]
                
                # 센서 데이터만 추출 (52개 센서)
                sensor_data = run_data.iloc[:, 3:].values  # faultNumber, simulationRun, sample 제외
                
                # 테스트 데이터에서 960→500 시점으로 평균 풀링 압축
                if is_test and sensor_data.shape[0] > 500:
                    sensor_data = average_pooling_1d(sensor_data, 500)
                
                # 라벨 (결함 번호)
                fault_label = run_data['faultNumber'].iloc[0]
                
                # 시계열 라벨링: 모든 시점에 동일한 라벨 적용
                time_labels = np.full(len(sensor_data), fault_label)
                
                self.data.append(sensor_data)
                self.labels.append(time_labels)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        # 데이터 정보 출력
        print(f"데이터 로드 완료: {len(self.data)}개 시뮬레이션 런")
        print(f"데이터 형태: {self.data.shape}")
        print(f"라벨 형태: {self.labels.shape}")
        
        if is_test:
            print(f"평균 풀링 압축 적용: 960 → 500 시점 (모든 시점 정보 활용)")
        
        # 클래스 분포 확인
        unique_labels = []
        for label_seq in self.labels:
            unique_labels.extend(label_seq)
        unique_labels = np.array(unique_labels)
        
        print(f"\n클래스 분포:")
        for i in range(max(unique_labels) + 1):
            count = np.sum(unique_labels == i)
            # 각 런의 첫 번째 시점 라벨로 런 개수 계산
            first_labels = np.array([label[0] for label in self.labels])
            runs = np.sum(first_labels == i)
            if i == 0:
                print(f"클래스 {i} (정상): {count:,} 샘플 ({runs}개 런)")
            else:
                print(f"클래스 {i} (결함{i}): {count:,} 샘플 ({runs}개 런)")
        
        # 특성 개수
        self.features_count = self.data.shape[2]  # 52개 센서
        self.class_count = len(np.unique(unique_labels))  # 클래스 개수
        
        print(f"\n데이터셋 정보:")
        print(f"  - 특성 개수: {self.features_count}")
        print(f"  - 클래스 개수: {self.class_count}")
        print(f"  - 시점 수: {self.data.shape[1]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'shot': self.data[idx],
            'label': self.labels[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


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
            sim_indices = batch["sim_idx"]
            
            # 모델 예측
            outputs = model(inputs)
            batch_preds = torch.argmax(outputs, dim=1)
            
            # CPU로 이동하고 numpy로 변환
            batch_preds = batch_preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            # 시뮬레이션별로 예측값 수집
            for pred, label, sim_idx in zip(batch_preds, batch_labels, sim_indices):
                if sim_idx not in predictions_by_sim:
                    predictions_by_sim[sim_idx] = []
                    labels_by_sim[sim_idx] = label[0]  # 각 시뮬레이션의 라벨은 동일
                predictions_by_sim[sim_idx].append(pred)
    
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
    
    run_predictions = []
    run_labels = []
    
    # 각 시뮬레이션 런의 예측 결과를 다수결로 결정
    for i in range(len(predictions)):
        # 각 런의 예측 결과 (500개 시점)
        run_pred = predictions[i]
        run_label = labels[i][0]  # 런의 실제 라벨 (모든 시점 동일)
        
        # 다수결로 런의 예측 라벨 결정
        unique, counts = np.unique(run_pred, return_counts=True)
        majority_pred = unique[np.argmax(counts)]
        
        run_predictions.append(majority_pred)
        run_labels.append(run_label)
    
    run_predictions = np.array(run_predictions)
    run_labels = np.array(run_labels)
    
    # 런 단위 정확도
    run_accuracy = accuracy_score(run_labels, run_predictions)
    total_correct = np.sum(run_predictions == run_labels)
    total_runs = len(run_labels)
    
    logger.info(f"\n=== 시뮬레이션 런 단위 평가 결과 ===")
    logger.info(f"총 정답: {total_correct}/{total_runs} = {run_accuracy:.4f}")
    
    # 실제 사용하는 클래스 확인 (0~12)
    unique_classes = np.unique(np.concatenate([run_labels, run_predictions]))
    max_class = max(unique_classes)
    num_classes = min(13, max_class + 1)  # 최대 13개 클래스 (0~12)
    
    # 런 단위 클래스별 성능
    run_precision, run_recall, run_f1, run_support = precision_recall_fscore_support(
        run_labels, run_predictions, average=None, zero_division=0
    )
    
    logger.info("\n=== 각 결함별 정답률 ===")
    correct_per_class = []
    for i in range(num_classes):  # 실제 사용하는 클래스만 출력
        fault_type = "정상" if i == 0 else f"결함{i}"
        correct_count = np.sum((run_labels == i) & (run_predictions == i))
        total_count = np.sum(run_labels == i)
        correct_per_class.append(correct_count)
        accuracy_rate = correct_count/total_count if total_count > 0 else 0
        logger.info(f"{fault_type:>6}: {correct_count:3d}/{total_count:3d} = {accuracy_rate:.3f}")
    
    # 전체 평균 성능 (실제 사용하는 클래스만)
    avg_precision = np.mean(run_precision[:num_classes])
    avg_recall = np.mean(run_recall[:num_classes])
    avg_f1 = np.mean(run_f1[:num_classes])
    
    logger.info(f"\n=== 전체 평균 성능 ===")
    logger.info(f"평균 정밀도: {avg_precision:.4f}")
    logger.info(f"평균 재현율: {avg_recall:.4f}")
    logger.info(f"평균 F1 점수: {avg_f1:.4f}")
    
    return {
        'run_accuracy': run_accuracy,
        'run_predictions': run_predictions,
        'run_labels': run_labels,
        'run_precision': run_precision[:num_classes],
        'run_recall': run_recall[:num_classes],
        'run_f1': run_f1[:num_classes],
        'run_support': run_support[:num_classes],
        'num_classes': num_classes,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'total_correct': total_correct,
        'total_runs': total_runs
    }


def plot_confusion_matrix(run_predictions, run_labels, save_path=None):
    """Run-level confusion matrix visualization"""
    logger = logging.getLogger(__name__)
    logger.info("런 단위 혼동 행렬 생성 중...")
    
    # 실제 사용하는 클래스 확인
    unique_classes = np.unique(np.concatenate([run_labels, run_predictions]))
    max_class = max(unique_classes)
    num_classes = min(13, max_class + 1)  # 최대 13개 클래스 (0~12)
    
    # 혼동 행렬 계산 (런 단위)
    cm = confusion_matrix(run_labels, run_predictions)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    class_names = ["Normal"] + [f"Fault{i}" for i in range(1, num_classes)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Tennessee Eastman Process Run-Level Fault Classification\n(Average Pooling: 960→500 timesteps)')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"혼동 행렬 저장: {save_path}")
    
    plt.show()


def save_results(results, save_dir):
    """Save run-level results"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save metrics
    metrics_file = os.path.join(save_dir, 'evaluation_metrics_run_level_avgpool.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Tennessee Eastman Process 런 단위 결함 탐지 평가 결과\n")
        f.write("평균 풀링 압축 적용: 960 → 500 시점 (모든 시점 정보 활용)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"총 정답: {results['total_correct']}/{results['total_runs']} = {results['run_accuracy']:.4f}\n\n")
        
        f.write("각 결함별 정답률:\n")
        for i in range(results['num_classes']):
            fault_type = "정상" if i == 0 else f"결함{i}"
            correct_count = np.sum((results['run_labels'] == i) & (results['run_predictions'] == i))
            total_count = np.sum(results['run_labels'] == i)
            accuracy_rate = correct_count/total_count if total_count > 0 else 0
            f.write(f"  {fault_type}: {correct_count:3d}/{total_count:3d} = {accuracy_rate:.3f}\n")
        
        f.write(f"\n전체 평균 성능:\n")
        f.write(f"  평균 정밀도: {results['avg_precision']:.4f}\n")
        f.write(f"  평균 재현율: {results['avg_recall']:.4f}\n")
        f.write(f"  평균 F1 점수: {results['avg_f1']:.4f}\n")
    
    # 런 단위 예측 결과 저장
    pred_file = os.path.join(save_dir, 'run_predictions_avgpool.npz')
    np.savez(pred_file, 
             run_predictions=results['run_predictions'], 
             run_labels=results['run_labels'])
    
    # 런 단위 혼동 행렬 저장
    cm_file = os.path.join(save_dir, 'confusion_matrix_run_level_avgpool.png')
    plot_confusion_matrix(results['run_predictions'], results['run_labels'], cm_file)
    
    logger.info(f"런 단위 평가 결과 저장 완료: {save_dir}")


@click.command()
@click.option('--model_path', required=True, type=str, help='학습된 discriminator 모델 경로 (.pth 파일)')
@click.option('--csv_dir', type=str, default='data/test_faults', help='CSV 파일들이 있는 디렉토리')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=16, help='배치 크기')
@click.option('--save_dir', type=str, default='evaluation_results_csv', help='결과 저장 디렉토리')
@click.option('--random_seed', type=int, default=42, help='랜덤 시드')
def main(model_path, csv_dir, cuda, batch_size, save_dir, random_seed):
    """
    Tennessee Eastman Process CSV 데이터 평가 (평균 풀링 압축 적용)
    
    테스트 데이터는 960 시점을 평균 풀링으로 500 시점으로 압축하여 평가합니다.
    모든 시점의 정보를 활용하여 성능을 측정합니다.
    
    사용법:
    python src/models/evaluate_model.py --model_path models/4_main_model/weights/199_epoch_discriminator.pth --cuda 0
    """
    
    # 로거 설정
    logger = setup_logger()
    logger.info("Tennessee Eastman Process CSV 데이터 평가 시작 (평균 풀링 압축 적용)")
    
    # 랜덤 시드 설정
    logger.info(f"Random Seed: {random_seed}")
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 디바이스 설정
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")
    
    # CSV 파일 목록 생성
    csv_files = [os.path.join(csv_dir, f"test_fault_{i}.csv") for i in range(13)]
    
    # 존재하는 파일만 필터링
    existing_files = [f for f in csv_files if os.path.exists(f)]
    logger.info(f"발견된 CSV 파일: {len(existing_files)}개")
    
    if not existing_files:
        logger.error("CSV 파일을 찾을 수 없습니다!")
        return
    
    # 데이터 변환 설정
    transform = transforms.Compose([
        CSVToTensor(),
        CSVNormalize()
    ])
    
    # 테스트 데이터셋 로드 (평균 풀링 압축 적용)
    logger.info("CSV 데이터셋 로드 중...")
    test_dataset = TEPCSVDatasetCompressed(existing_files, transform=transform, is_test=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 오류 방지
        drop_last=False
    )
    
    logger.info(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 평가
    accuracy, predictions, labels = evaluate_model(model, test_loader, device)
    
    # 결과 분석
    results = analyze_results(predictions, labels, logger)
    
    # 결과 저장
    save_results(results, save_dir)
    
    logger.info("CSV 데이터 평가 완료! (평균 풀링 압축 적용)")


if __name__ == '__main__':
    main() 