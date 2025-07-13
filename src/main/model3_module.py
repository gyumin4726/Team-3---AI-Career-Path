"""
Model3 (반응 변수 예측) 모듈
Tennessee Eastman Process의 반응 변수 예측 기능
"""

import numpy as np
import torch
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# 상위 디렉토리 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model3'))
from src.model3.model3 import TCNSeq2Seq


class Model3Module:
    """
    Model3 (TCNSeq2Seq) 모듈
    반응 변수 예측
    """
    
    def __init__(self):
        """Model3 모듈 초기화"""
        self.model3 = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {
            'predicted_x': None,
            'fault_time': None,
            'context_size': None
        }
        
        # 하이퍼파라미터 (train_model3.py와 동일)
        self.x_dim = 41  # 반응 변수 개수 (0~40번째 변수)
        self.m_dim = 11  # 조작 변수 개수 (41~51번째 변수)
        self.c_lat = 128  # 잠재 차원
        
        self.load_model3()
    
    def load_model3(self):
        """Model3 (TCNSeq2Seq) 로드 - evaluate_model3.py 기반"""
        try:
            # 사전 학습된 가중치 경로 (evaluate_model3.py와 동일)
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_pretrained', 'model3', '10_epoch_checkpoint.pth')
            
            # Model3 생성 및 로드 (evaluate_model3.py와 동일)
            self.model3 = TCNSeq2Seq(x_dim=self.x_dim, m_dim=self.m_dim, c_lat=self.c_lat).to(self.device)
            
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model3.load_state_dict(checkpoint['model_state_dict'])
                self.model3.eval()
                print(f"Model3 로드 성공 (epoch {checkpoint['epoch']})")
            else:
                print(f"Model3 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
                self.model3 = None
                
        except Exception as e:
            print(f"Model3 로드 실패: {e}")
            self.model3 = None
    
    def predict_response_variables(self, data_sequence: np.ndarray, 
                                 fault_time: int) -> np.ndarray:
        """
        반응 변수 예측 (Model3) - evaluate_model3.py와 동일한 로직
        
        Args:
            data_sequence: 전체 데이터 시퀀스 (B, 50, 52) - evaluate_model3.py와 동일
            fault_time: Model1에서 감지된 fault 시점 (슬라이딩 윈도우 인덱스)
            
        Returns:
            predicted_data: 예측된 전체 데이터 시퀀스 (B, 50, 52)
        """
        print("Model3: 반응 변수 예측")
        print(f"입력: 전체 데이터 시퀀스 형태={data_sequence.shape}")
        print(f"Fault 시점: {fault_time} (슬라이딩 윈도우 인덱스)")
        
        if self.model3 is None:
            print("Model3이 로드되지 않았습니다. 임시 결과를 사용합니다.")
            # 전체 데이터 반환
            self.results['predicted_x'] = data_sequence.copy()
            self.results['fault_time'] = fault_time
            return self.results['predicted_x']
        
        try:
            # evaluate_model3.py와 동일한 방식으로 처리
            batch_size = 92  # evaluate_model3.py와 동일
            num_simulations = len(data_sequence) // batch_size
            
            print(f"처리할 시뮬레이션 수: {num_simulations}")
            print(f"시뮬레이션당 윈도우 수: {batch_size}")
            
            predicted_x_list = []
            
            for sim_idx in range(num_simulations):
                print(f"시뮬레이션 {sim_idx + 1}/{num_simulations} 처리 중...")
                
                # 현재 시뮬레이션 데이터 추출 (evaluate_model3.py와 동일)
                start_idx = sim_idx * batch_size
                end_idx = start_idx + batch_size
                sim_data = data_sequence[start_idx:end_idx]  # (92, 50, 52)
                
                # 변수 분할 (evaluate_model3.py와 동일)
                x_data = sim_data[:, :, :self.x_dim]  # (92, 50, 41) - 반응 변수
                m_data = sim_data[:, :, self.x_dim:]  # (92, 50, 11) - 조작 변수
                
                # fault_time을 배치 인덱스로 변환 (evaluate_model3.py와 동일)
                batch_idx = fault_time // 50
                
                # fault_time이 현재 시뮬레이션 범위 내인지 확인
                if batch_idx < len(sim_data):
                    # 컨텍스트: 배치 0~batch_idx (현재 배치까지 포함) - evaluate_model3.py와 동일
                    x_ctx = x_data[:batch_idx+1, :, :]  # 이전 배치들 + 현재 배치
                    m_ctx = m_data[:batch_idx+1, :, :]
                    
                    # 미래: 배치 batch_idx+1부터 끝까지 - evaluate_model3.py와 동일
                    if batch_idx + 1 < len(sim_data):
                        m_fut = m_data[batch_idx+1:, :, :]  # 이후 배치들
                    else:
                        # 현재 배치가 마지막인 경우 빈 텐서 생성
                        m_fut = np.empty((0, m_data.shape[1], m_data.shape[2]))
                    
                    # Model3 예측 (evaluate_model3.py와 동일)
                    with torch.no_grad():
                        x_ctx_tensor = torch.FloatTensor(x_ctx).to(self.device)
                        m_ctx_tensor = torch.FloatTensor(m_ctx).to(self.device)
                        m_fut_tensor = torch.FloatTensor(m_fut).to(self.device)
                        
                        # 예측
                        x_fut_pred = self.model3(x_ctx_tensor, m_ctx_tensor, m_fut_tensor)
                        x_fut_pred = x_fut_pred.cpu().numpy()

                    # 새로운 시퀀스 생성 (evaluate_model3.py와 동일)
                    # fault_time 이전: 원본 X
                    # fault_time 이후: 예측된 X'
                    
                    # 컨텍스트와 예측 결과를 연결
                    if x_fut_pred.shape[0] > 0:  # 예측 결과가 있는 경우
                        new_x_sequence = np.concatenate([x_ctx, x_fut_pred], axis=0)  # (total_batches, 50, 41)
                    else:
                        new_x_sequence = x_ctx  # 예측할 미래가 없는 경우
                    
                    # 원본 형태로 복원 (92, 50, 41) - evaluate_model3.py와 동일
                    if new_x_sequence.shape[0] < 92:
                        # 부족한 배치는 마지막 배치로 채움
                        last_batch = new_x_sequence[-1:].repeat(92 - new_x_sequence.shape[0], axis=0)
                        new_x_sequence = np.concatenate([new_x_sequence, last_batch], axis=0)
                    elif new_x_sequence.shape[0] > 92:
                        # 초과하는 배치는 제거
                        new_x_sequence = new_x_sequence[:92]
                    
                    # 새로운 데이터 생성 (X' + M) - evaluate_model3.py와 동일
                    new_sim_data = np.concatenate([new_x_sequence, m_data], axis=2)  # (92, 50, 52)
                    
                    predicted_x_list.append(new_sim_data)
                else:
                    # fault_time이 범위를 벗어난 경우 원본 데이터 사용
                    predicted_x_list.append(sim_data)
            
            # 결과 합치기 (evaluate_model3.py와 동일)
            predicted_data = np.concatenate(predicted_x_list, axis=0)
            
            # 결과 저장 (전체 데이터 반환)
            self.results['predicted_x'] = predicted_data
            self.results['fault_time'] = fault_time
            
            print(f"결과: 예측된 전체 시퀀스 형태={predicted_data.shape}")
            return predicted_data
            
        except Exception as e:
            print(f"Model3 예측 중 오류: {e}")
            # 오류 발생 시 원본 데이터 반환
            self.results['predicted_x'] = data_sequence.copy()
            self.results['fault_time'] = fault_time
            return self.results['predicted_x']
    
    def get_results(self) -> Dict[str, Any]:
        """
        Model3 결과 반환
        
        Returns:
            Model3 결과 딕셔너리
        """
        return {
            'predicted_x': self.results['predicted_x'],
            'fault_time': self.results['fault_time']
        }
    
    def is_loaded(self) -> bool:
        """
        Model3이 로드되었는지 확인
        
        Returns:
            로드되었으면 True, 아니면 False
        """
        return self.model3 is not None 