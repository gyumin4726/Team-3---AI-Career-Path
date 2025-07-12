import os
import torch
import numpy as np
import click
import logging
from pathlib import Path
import sys
from src.data.dataset import TEPNPYDataset, CSVToTensor
from src.model3.model3 import TCNSeq2Seq

"""Model3 평가 및 예측 결과 NPY 파일 생성"""

@click.command()
@click.option('--cuda', type=int, default=0)
@click.option('--checkpoint_path', type=str, required=True, help='Model3 체크포인트 파일 경로')
@click.option('--test_data_path', type=str, default='data/test_X_model1.npy', help='테스트 데이터 경로')
@click.option('--test_labels_path', type=str, default='data/test_Y_model1.npy', help='테스트 라벨 경로')
@click.option('--output_dir', type=str, default='data/model3_results', help='결과 저장 디렉토리')
def main(cuda, checkpoint_path, test_data_path, test_labels_path, output_dir):
    """
    Model3 평가 및 예측 결과 생성
    """
    # 로그 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Evaluation begin on {device}')

    # 하이퍼파라미터 (train_model3.py와 동일)
    x_dim = 41  # 반응 변수 개수 (0~40번째 변수)
    m_dim = 11  # 조작 변수 개수 (41~51번째 변수)
    c_lat = 128  # 잠재 차원

    # Model3 로드
    logger.info(f'Loading Model3 from: {checkpoint_path}')
    netM3 = TCNSeq2Seq(x_dim=x_dim, m_dim=m_dim, c_lat=c_lat).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netM3.load_state_dict(checkpoint['model_state_dict'])
    netM3.eval()
    logger.info(f'Model3 loaded successfully (epoch {checkpoint["epoch"]})')

    # 테스트 데이터 로드
    logger.info(f'Loading test data from: {test_data_path}')
    test_data = np.load(test_data_path)
    test_labels = np.load(test_labels_path)
    logger.info(f'Test data shape: {test_data.shape}')
    logger.info(f'Test labels shape: {test_labels.shape}')

    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 예측 결과 저장용 배열
    predicted_data = []
    original_labels = []

    # 배치 단위로 처리 (한 시뮬레이션 = 92개 윈도우)
    batch_size = 92  # 테스트 데이터는 92개 윈도우
    num_simulations = len(test_data) // batch_size
    
    logger.info(f'Test data processing:')
    logger.info(f'  - Total windows: {len(test_data)}')
    logger.info(f'  - Windows per simulation: {batch_size}')
    logger.info(f'  - Number of simulations: {num_simulations}')
    
    logger.info(f'Processing {num_simulations} simulations...')

    for sim_idx in range(num_simulations):
        logger.info(f'Processing simulation {sim_idx + 1}/{num_simulations}')
        
        # 현재 시뮬레이션 데이터 추출
        start_idx = sim_idx * batch_size
        end_idx = start_idx + batch_size
        sim_data = test_data[start_idx:end_idx]  # (92, 50, 52)
        sim_labels = test_labels[start_idx:end_idx]  # (92,)

        # 변수 분할
        x_data = sim_data[:, :, :x_dim]  # (92, 50, 41) - 반응 변수
        m_data = sim_data[:, :, x_dim:]  # (92, 50, 11) - 조작 변수

        # fault_time 설정 (임시로 중간 시점 사용)
        fault_time = 1000  # 실제로는 Model1에서 받음
        batch_idx = fault_time // 50

        # fault_time이 현재 시뮬레이션 범위 내인지 확인
        if batch_idx < len(sim_data):
            # 컨텍스트: 배치 0~batch_idx (현재 배치까지 포함)
            x_ctx = x_data[:batch_idx+1, :, :]  # 이전 배치들 + 현재 배치
            m_ctx = m_data[:batch_idx+1, :, :]
            
            # 미래: 배치 batch_idx+1부터 끝까지
            if batch_idx + 1 < len(sim_data):
                x_fut_gt = x_data[batch_idx+1:, :, :]  # 이후 배치들
                m_fut = m_data[batch_idx+1:, :, :]  # 이후 배치들
            else:
                # 현재 배치가 마지막인 경우 빈 텐서 생성
                x_fut_gt = np.empty((0, x_data.shape[1], x_data.shape[2]))
                m_fut = np.empty((0, m_data.shape[1], m_data.shape[2]))

            # Model3 예측
            with torch.no_grad():
                x_ctx_tensor = torch.FloatTensor(x_ctx).to(device)
                m_ctx_tensor = torch.FloatTensor(m_ctx).to(device)
                m_fut_tensor = torch.FloatTensor(m_fut).to(device)
                
                # 예측
                x_fut_pred = netM3(x_ctx_tensor, m_ctx_tensor, m_fut_tensor)
                x_fut_pred = x_fut_pred.cpu().numpy()

            # 새로운 시퀀스 생성
            # fault_time 이전: 원본 X
            # fault_time 이후: 예측된 X'
            
            # 컨텍스트와 예측 결과를 연결
            if x_fut_pred.shape[0] > 0:  # 예측 결과가 있는 경우
                new_x_sequence = np.concatenate([x_ctx, x_fut_pred], axis=0)  # (total_batches, 50, 41)
            else:
                new_x_sequence = x_ctx  # 예측할 미래가 없는 경우
            
            # 원본 형태로 복원 (92, 50, 41)
            # 92개 배치로 맞춤
            if new_x_sequence.shape[0] < 92:
                # 부족한 배치는 마지막 배치로 채움
                last_batch = new_x_sequence[-1:].repeat(92 - new_x_sequence.shape[0], axis=0)
                new_x_sequence = np.concatenate([new_x_sequence, last_batch], axis=0)
            elif new_x_sequence.shape[0] > 92:
                # 초과하는 배치는 제거
                new_x_sequence = new_x_sequence[:92]
            
            # 새로운 데이터 생성 (X' + M)
            new_sim_data = np.concatenate([new_x_sequence, m_data], axis=2)  # (92, 50, 52)
            
            predicted_data.append(new_sim_data)
            original_labels.append(sim_labels)
            
        else:
            # fault_time이 범위를 벗어난 경우 원본 데이터 사용
            predicted_data.append(sim_data)
            original_labels.append(sim_labels)

    # 결과 합치기
    predicted_data = np.concatenate(predicted_data, axis=0)
    original_labels = np.concatenate(original_labels, axis=0)

    # NPY 파일 저장
    output_x_path = os.path.join(output_dir, 'model3_predicted_X.npy')
    output_y_path = os.path.join(output_dir, 'model3_predicted_Y.npy')
    
    np.save(output_x_path, predicted_data)
    np.save(output_y_path, original_labels)
    
    logger.info(f'Results saved:')
    logger.info(f'  - Predicted X: {output_x_path} (shape: {predicted_data.shape})')
    logger.info(f'  - Original Y: {output_y_path} (shape: {original_labels.shape})')
    
    # 통계 정보 출력
    logger.info(f'Statistics:')
    logger.info(f'  - Original data range: {test_data.min():.4f} ~ {test_data.max():.4f}')
    logger.info(f'  - Predicted data range: {predicted_data.min():.4f} ~ {predicted_data.max():.4f}')
    logger.info(f'  - Labels: {np.unique(original_labels)}')

    logger.info('Evaluation completed successfully!')

if __name__ == '__main__':
    main() 