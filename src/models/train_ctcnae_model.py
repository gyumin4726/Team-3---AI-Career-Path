#!/usr/bin/env python3
"""
Tennessee Eastman Process 정상화 C-TCN-AE 모델 학습 스크립트
고장 데이터를 진짜 정상으로 재구성하도록 학습
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import click
import logging
from datetime import datetime

from src.data.dataset import TEPNPYDataset, CSVToTensor
from src.models.ctcnae_models import ConditionalTCNAutoencoder


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, normal_dataset, optimizer, device):
    """한 에포크 학습 - 고장→정상 정상화 모델"""
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(train_loader):
        # 고장 데이터 (입력)
        m_seq = batch["shot"].to(device)  # [B, 50, 11] - 고장 데이터
        fault_labels = batch["label"].to(device)  # [B,] - 고장 라벨
        batch_size = m_seq.size(0)  # 현재 배치 크기
        
        # 고장 데이터의 인덱스를 기반으로 정상 데이터 가져오기
        # 고장 데이터셋의 인덱스를 정상 데이터셋의 인덱스로 변환
        fault_indices = batch.get("index", torch.arange(len(batch["shot"])))  # 배치 내 인덱스
        
        # 정상 데이터 가져오기 (고장과 같은 순서)
        normal_seqs = []
        for idx in fault_indices:
            # 고장 인덱스를 정상 인덱스로 변환 (같은 시뮬레이션)
            normal_idx = idx % len(normal_dataset)  # 정상 데이터셋 크기로 모듈로
            normal_data = normal_dataset[normal_idx]
            normal_seqs.append(normal_data["shot"])
        
        normal_seq = torch.stack(normal_seqs).to(device)  # [B, 50, 11] - 정상 데이터
        normal_labels = torch.zeros(batch_size, dtype=torch.long).to(device)  # [B,] - 정상 라벨 (모두 0)
        
        # 첫 배치의 shape 출력
        if i == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"고장 입력 shape: {m_seq.shape}")
            logger.info(f"고장 라벨 shape: {fault_labels.shape}")
            logger.info(f"고장 라벨 고유값: {torch.unique(fault_labels).cpu().numpy()}")
            logger.info(f"정상 타겟 shape: {normal_seq.shape}")
            logger.info(f"정상 라벨 shape: {normal_labels.shape}")
            logger.info(f"정상 라벨 고유값: {torch.unique(normal_labels).cpu().numpy()}")
        
        # 정상화 모델: 고장 데이터를 정상으로 재구성
        normal_condition = torch.zeros_like(fault_labels)  # 모든 조건을 정상(0)으로
        m_rec = model(m_seq, normal_condition)
        
        # Loss 계산 (MSE) - 고장 데이터를 진짜 정상으로 재구성하는 성능
        loss = torch.nn.functional.mse_loss(m_rec, normal_seq)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


@click.command()
@click.option('--train_data', type=str, default='data/train_X_model2.npy', help='훈련 데이터 NPY 파일 경로')
@click.option('--train_labels', type=str, default='data/train_Y_model2.npy', help='훈련 라벨 NPY 파일 경로')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=128, help='배치 크기')
@click.option('--epochs', type=int, default=100, help='학습 에포크 수')
@click.option('--lr', type=float, default=0.001, help='학습률')
@click.option('--channels', type=str, default='32,64,64', help='TCN 채널 크기 (콤마로 구분)')
@click.option('--kernel_size', type=int, default=3, help='TCN 커널 크기')
def main(train_data, train_labels, cuda, batch_size, epochs, lr, channels, kernel_size):
    """
    정상화 C-TCN-AE 모델 학습 메인 함수
    고장 데이터를 진짜 정상으로 재구성하도록 학습
    """
    logger = setup_logger()
    
    # 저장 디렉토리 설정
    save_dir = f"model_pretrained/model2"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("model_pretrained", exist_ok=True)
    os.makedirs("model_pretrained/model2", exist_ok=True)
    
    # GPU 설정
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"장치: {device}")
    
    # 전체 데이터셋 생성
    full_dataset = TEPNPYDataset(
        data_path=train_data,
        labels_path=train_labels,
        transform=CSVToTensor(),
        is_test=False
    )
    
    # 정상 데이터만 필터링
    normal_indices = [i for i, label in enumerate(full_dataset.labels) if label == 0]
    normal_dataset = torch.utils.data.Subset(full_dataset, normal_indices)
    
    # 고장 데이터만 필터링 (정상 제외)
    fault_indices = [i for i, label in enumerate(full_dataset.labels) if label > 0]
    fault_dataset = torch.utils.data.Subset(full_dataset, fault_indices)
    
    logger.info(f"\n데이터셋 정보:")
    logger.info(f"전체 데이터셋 크기: {len(full_dataset)}")
    logger.info(f"정상 데이터셋 크기: {len(normal_dataset)}")
    logger.info(f"고장 데이터셋 크기: {len(fault_dataset)}")
    
    # 데이터 로더 생성
    fault_loader = DataLoader(
        fault_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    # 채널 리스트 파싱
    channels = [int(c) for c in channels.split(',')]
    
    # 모델 생성 (정상화 모델)
    model = ConditionalTCNAutoencoder(
        m_dim=11,  # 측정값만 사용
        fault_dim=13,  # fault 타입 수
        channels=channels,
        kernel_size=kernel_size
    ).to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 학습 루프
    best_train_loss = float('inf')
    for epoch in range(epochs):
        # 학습
        train_loss = train_epoch(model, fault_loader, normal_dataset, optimizer, device)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss:.6f}")
        
        # 모델 저장 (학습 손실이 개선된 경우)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            checkpoint_path = os.path.join(save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            logger.info(f"모델 저장됨: {checkpoint_path}")
        
        # 주기적으로 체크포인트 저장 (10 에포크마다)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"{epoch+1}_epoch_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            logger.info(f"체크포인트 저장됨: {checkpoint_path}")
    
    logger.info("정상화 모델 학습 완료!")


if __name__ == '__main__':
    main() 