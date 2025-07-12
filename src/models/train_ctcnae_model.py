#!/usr/bin/env python3
"""
Tennessee Eastman Process C-TCN-AE 모델 학습 스크립트
Conditional Temporal Convolutional Network Autoencoder를 사용하여 조건부 시퀀스 학습
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


def train_epoch(model, train_loader, optimizer, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(train_loader):
        # 측정값(m)과 라벨 사용
        m_seq = batch["shot"].to(device)  # [B, 50, 11] - 측정값만 사용
        fault_labels = batch["label"].to(device)  # [B,] - 라벨
        
        # 첫 배치의 shape 출력
        if i == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"m_seq shape: {m_seq.shape}")  # [B, 50, 11]
            logger.info(f"fault_labels shape: {fault_labels.shape}")  # [B,]
            logger.info(f"fault_labels 고유값: {torch.unique(fault_labels).cpu().numpy()}")
        
        # Forward pass (조건부 모델)
        m_rec = model(m_seq, fault_labels)
        
        # Loss 계산 (MSE)
        loss = torch.nn.functional.mse_loss(m_rec, m_seq)
        
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
    C-TCN-AE 모델 학습 메인 함수
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
    
    # 데이터셋 생성 (모든 fault 타입 포함)
    train_dataset = TEPNPYDataset(
        data_path=train_data,
        labels_path=train_labels,
        transform=CSVToTensor(),
        is_test=False  # 50 윈도우 크기 사용
    )
    
    # 데이터셋 정보 출력
    logger.info("\n데이터셋 정보:")
    first_data = train_dataset[0]
    logger.info(f"입력 데이터 shape: {first_data['shot'].shape}")  # [50, 52]
    logger.info(f"측정값(m) shape: {first_data['shot'][:, :11].shape}")  # [50, 11]
    logger.info(f"라벨: {first_data['label']}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 첫 번째 배치 shape 출력
    first_batch = next(iter(train_loader))
    logger.info("\n첫 번째 배치 정보:")
    logger.info(f"배치 전체 shape: {first_batch['shot'].shape}")  # [B, 50, 52]
    logger.info(f"배치 측정값(m) shape: {first_batch['shot'][:, :, :11].shape}")  # [B, 50, 11]
    logger.info(f"배치 라벨 shape: {first_batch['label'].shape}")  # [B,]
    logger.info(f"라벨 고유값: {torch.unique(first_batch['label']).numpy()}")
    
    # 채널 리스트 파싱
    channels = [int(c) for c in channels.split(',')]
    
    # 모델 생성 (조건부 모델)
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
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
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
    
    logger.info("학습 완료!")


if __name__ == '__main__':
    main() 