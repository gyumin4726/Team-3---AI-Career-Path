import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import click
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import sys
from src.data.dataset import TEPNPYDataset, CSVToTensor
from src.model1.utils import get_latest_model_id
from src.model3.model3 import TCNSeq2Seq
import torch.backends.cudnn as cudnn
import random

"""This is training of TCNSeq2Seq model for Model3."""

@click.command()
@click.option('--cuda', type=int, default=0)
@click.option('--random_seed', type=int, default=42)
@click.option('--resume_from', type=str, default=None, help='체크포인트 파일 경로')
def main(cuda, random_seed, resume_from):
    """
    Model3 (TCNSeq2Seq) 모델 훈련 - NPY 데이터 사용
    """
    # 로그 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Training begin on {device}')

    # 랜덤 시드 설정
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = True

    # 하이퍼파라미터
    bs = 46  # 한 시뮬레이션 전체 (46개 윈도우)
    x_dim = 41  # 반응 변수 개수 (0~40번째 변수)
    m_dim = 11  # 조작 변수 개수 (41~51번째 변수)
    c_lat = 128  # 잠재 차원
    ctx_windows = 30  # 컨텍스트 윈도우 개수
    checkpoint_every = 5
    epochs = 200
    
    # NPY 파일 경로 설정
    train_data_path = "data/train_X_model1.npy"
    train_labels_path = "data/train_Y_model1.npy"
    
    # 데이터 shape 확인
    logger.info(f"데이터 로딩 중...")
    temp_data = np.load(train_data_path)
    logger.info(f"train_X_model1.npy shape: {temp_data.shape}")
    logger.info(f"변수 분할: 반응 변수 {x_dim}개 (0~{x_dim-1}), 조작 변수 {m_dim}개 ({x_dim}~{x_dim+m_dim-1})")

    # 데이터 로더 설정
    transform = CSVToTensor()
    trainset = TEPNPYDataset(
        data_path=train_data_path,
        labels_path=train_labels_path,
        transform=transform,
        is_test=False
    )
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    logger.info("Dataset loaded successfully")

    # Model3: TCNSeq2Seq 모델
    netM3 = TCNSeq2Seq(x_dim=x_dim, m_dim=m_dim, c_lat=c_lat).to(device)
    optimizerM3 = optim.Adam(netM3.parameters(), lr=0.0001)

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f'Loading checkpoint: {resume_from}')
        checkpoint = torch.load(resume_from)
        netM3.load_state_dict(checkpoint['model_state_dict'])
        optimizerM3.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resuming from epoch {start_epoch}')

    logger.info("Model3 initialized successfully")

    # 손실 함수
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    # 모델 저장 디렉토리 생성
    os.makedirs("model_pretrained/model3", exist_ok=True)  # 디렉토리 미리 생성
    
    for epoch in range(start_epoch, epochs):
        logger.info(f'Epoch {epoch}/{epochs} training...')
        netM3.train()

        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for i, data in enumerate(trainloader, 0):
            # 데이터 준비
            full_inputs = data["shot"].to(device)  # (B, 50, 52)

            # x와 m 분리 (TEP 데이터 구조에 맞춰 분리)
            # 반응 변수: 0~40번째 변수 (41개)
            # 조작 변수: 41~51번째 변수 (11개)
            x_inputs = full_inputs[:, :, :x_dim]  # (B, 50, 41) - 반응 변수
            m_inputs = full_inputs[:, :, x_dim:]  # (B, 50, 11) - 조작 변수
            
            # 데이터 분할 검증 (첫 번째 배치에서만)
            if i == 0:
                logger.info(f"데이터 분할 검증:")
                logger.info(f"  - 전체 입력: {full_inputs.shape}")
                logger.info(f"  - 반응 변수 (x): {x_inputs.shape}")
                logger.info(f"  - 조작 변수 (m): {m_inputs.shape}")

                logger.info(f"  - 분할 검증: {x_inputs.shape[2]} + {m_inputs.shape[2]} = {full_inputs.shape[2]}")

            # fault_time 기반 동적 분할 (훈련 시에는 랜덤 시점 사용)
            fault_time = np.random.randint(500, 1800)  # 훈련 시 랜덤 값 (실제로는 Model1에서 받음)
            batch_idx = fault_time // 50  # 어느 배치에 속하는지
            timestep_in_batch = fault_time % 50  # 배치 내 시점
            
            # 컨텍스트: fault_time 이전 모든 시점
            # 미래: fault_time 이후 모든 시점
            if batch_idx < x_inputs.shape[0]:  # 배치 범위 내
                # 컨텍스트: 배치 0~batch_idx (현재 배치까지 포함)
                x_ctx = x_inputs[:batch_idx+1, :, :]  # 이전 배치들 + 현재 배치
                m_ctx = m_inputs[:batch_idx+1, :, :]
                
                # 미래: 배치 batch_idx+1부터 끝까지
                # X만 예측, M은 그대로 사용 (조작 변수는 정상화되지 않음)
                if batch_idx + 1 < x_inputs.shape[0]:
                    x_fut_gt = x_inputs[batch_idx+1:, :, :]  # 이후 배치들
                    m_fut = m_inputs[batch_idx+1:, :, :]  # 이후 배치들
                else:
                    # 현재 배치가 마지막인 경우 빈 텐서 생성
                    x_fut_gt = torch.empty(0, x_inputs.shape[1], x_inputs.shape[2], device=x_inputs.device)
                    m_fut = torch.empty(0, m_inputs.shape[1], m_inputs.shape[2], device=m_inputs.device)

            else:
                # fault_time이 범위를 벗어난 경우 기본값 사용
                x_ctx = x_inputs[:, :ctx_windows, :]
                m_ctx = m_inputs[:, :ctx_windows, :]
                m_fut = m_inputs[:, ctx_windows:, :]
                x_fut_gt = x_inputs[:, ctx_windows:, :]

            netM3.zero_grad()

            # Forward pass
            # Model3: 과거 X + 과거 M + 미래 M → 미래 X 예측
            x_fut_pred = netM3(x_ctx, m_ctx, m_fut)  # (B, T, 41) - 반응 변수만 예측

            # 손실 계산 (가중치 적용)
            mse_loss = 10.0 * mse_criterion(x_fut_pred, x_fut_gt)
            mae_loss = 1.0 * mae_criterion(x_fut_pred, x_fut_gt)
            
            # 전체 손실 (MSE + MAE)
            total_loss_batch = mse_loss + mae_loss
            
            total_loss_batch.backward()
            optimizerM3.step()

            total_loss += mse_loss.item()
            total_mae += mae_loss.item()
            num_batches += 1

            # 로깅
            if i % 50 == 0:
                logger.info(f'[{epoch}/{epochs}][{i}/{len(trainloader)}] MSE: {mse_loss.item():.4f} MAE: {mae_loss.item():.4f}')

        # 에폭 평균 손실
        avg_mse = total_loss / num_batches
        avg_mae = total_mae / num_batches
        logger.info(f'Epoch {epoch} - Avg MSE: {avg_mse:.4f}, Avg MAE: {avg_mae:.4f}')

        # 체크포인트 저장
        if (epoch % checkpoint_every == 0) or (epoch == (epochs - 1)):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': netM3.state_dict(),
                'optimizer_state_dict': optimizerM3.state_dict(),
                'avg_mse': avg_mse,
                'avg_mae': avg_mae,
            }
            checkpoint_path = f"model_pretrained/model3/{epoch}_epoch_checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')

    logger.info(f'Training completed for {epochs} epochs.')

if __name__ == '__main__':
    main()