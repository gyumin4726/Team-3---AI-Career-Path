"""
Model1 (Fault 탐지 + 분류) 모듈
Tennessee Eastman Process의 fault 시점 탐지 및 분류 기능
"""

import numpy as np
import torch
import sys
import os
from collections import Counter
from typing import Dict, List, Tuple, Any

# 상위 디렉토리 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model1'))
from src.model1.convolutional_models import CNN1D2DDiscriminatorMultitask
from src.model1.evaluate_model import detect_fault_onset, load_model as load_model1_from_checkpoint


class Model1Module:
    """
    Model1 (CNN1D2D Discriminator) 모듈
    Fault 시점 탐지 + Fault 종류 분류
    """
    
    def __init__(self):
        """Model1 모듈 초기화"""
        self.model1 = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {
            'fault_time': None,
            'fault_class': None,
            'original_fault_time': None  # 원본 시점 추가
        }
        self.load_model1()
    
    def convert_window_index_to_original_time(self, window_index: int, window_size: int = 50, step_size: int = 10) -> int:
        """
        슬라이딩 윈도우 시점을 원본 시점으로 변환
        
        Args:
            window_index: 슬라이딩 윈도우 시점 (0~4599)
            window_size: 윈도우 크기 (기본값: 50)
            step_size: 스텝 크기 (기본값: 10)
            
        Returns:
            original_time: 원본 시점 (0~959)
        """
        # 슬라이딩 윈도우 시점을 원본 시점으로 변환
        # window_index는 전체 시점 중의 인덱스 (0~4599)
        # 이를 원본 시점 (0~959)으로 변환
        
        # 윈도우 번호와 윈도우 내 시점 계산
        window_num = window_index // window_size
        timestep_in_window = window_index % window_size
        
        # 원본 시점 계산: 윈도우 시작 시점 + 윈도우 내 시점
        original_time = window_num * step_size + timestep_in_window
        
        # 원본 데이터 범위(0~959)를 벗어나지 않도록 제한
        original_time = min(original_time, 959)
        
        return original_time
    
    def load_model1(self):
        """Model1 (CNN1D2D Discriminator) 로드"""
        try:
            # 사전 학습된 가중치 경로
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_pretrained', 'model1', '30_epoch_checkpoint.pth')
            
            # evaluate_model.py의 load_model 함수 사용
            self.model1 = load_model1_from_checkpoint(checkpoint_path, self.device)
            
        except Exception as e:
            print(f"Model1 로드 실패: {e}")
            self.model1 = None
    
    def detect_fault(self, data_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Fault 시점 탐지 + Fault 종류 분류
        Args:
            data_sequence: 전체 데이터 시퀀스 (B, 50, 52)
        Returns:
            {
                'is_normal': bool,
                'fault_class': str or None,
                'fault_time': int or None,  # 슬라이딩 윈도우 인덱스
                'original_fault_time': int or None  # 원본 시점
            }
        """
        print("Model1: Fault 시점 탐지 + Fault 종류 분류")
        print(f"입력 데이터 형태: {data_sequence.shape}")
        
        if self.model1 is None:
            print("Model1이 로드되지 않았습니다. 임시 결과를 사용합니다.")
            self.results['fault_time'] = 300
            self.results['fault_class'] = "fault_1"
            self.results['original_fault_time'] = self.convert_window_index_to_original_time(300)
            return {
                'is_normal': False,
                'fault_class': self.results['fault_class'],
                'fault_time': self.results['fault_time'],
                'original_fault_time': self.results['original_fault_time']
            }
        
        try:
            total_timesteps = data_sequence.shape[0] * data_sequence.shape[1]  # B × 50
            print(f"전체 시점 수: {total_timesteps}")
            batch_size = data_sequence.shape[0]
            fault_predictions = []
            detailed_predictions = []
            print(f"배치 크기: {batch_size}")
            for batch_idx in range(batch_size):
                batch_data = torch.FloatTensor(data_sequence[batch_idx]).unsqueeze(0).to(self.device)  # (1, 50, 52)
                with torch.no_grad():
                    type_logits, _ = self.model1(batch_data, None)
                    type_logits = type_logits.transpose(1, 2)
                    batch_predictions = torch.argmax(type_logits, dim=1).cpu().numpy()  # (50,)
                    detailed_predictions.extend(batch_predictions)
                    batch_majority = Counter(batch_predictions).most_common(1)[0][0]
                    fault_predictions.append(batch_majority)
            print(f"총 {len(fault_predictions)}개 배치 분석 완료")
            print(f"상세 예측: {len(detailed_predictions)}개 시점 분석 완료")
            prediction_counts = Counter(fault_predictions)
            print(f"배치별 예측 분포: {dict(prediction_counts)}")
            most_common_fault = prediction_counts.most_common(1)[0][0]
            fault_class = f"fault_{most_common_fault}" if most_common_fault != 0 else "normal"
            detailed_fault_time, _, correct_preds = detect_fault_onset(
                detailed_predictions, 
                most_common_fault, 
                threshold=5
            )
            original_fault_time = self.convert_window_index_to_original_time(detailed_fault_time)
            self.results['fault_time'] = detailed_fault_time
            self.results['original_fault_time'] = original_fault_time
            self.results['fault_class'] = fault_class
            print(f"Model1 결과:")
            print(f"  - 예측된 fault 클래스: {fault_class}")
            print(f"  - 슬라이딩 윈도우 인덱스: {detailed_fault_time} / {total_timesteps}")
            print(f"  - 원본 시점: {original_fault_time} / 959")
            if correct_preds:
                print(f"  - 연속 정답 예측: {correct_preds}")
            if detailed_fault_time > 0:
                batch_num = detailed_fault_time // 50
                timestep_in_batch = detailed_fault_time % 50
                print(f"  - 상세 위치: 배치 {batch_num}, 배치 내 시점 {timestep_in_batch}")
        except Exception as e:
            print(f"Model1 추론 중 오류: {e}")
            self.results['fault_time'] = 300
            self.results['fault_class'] = "fault_1"
            self.results['original_fault_time'] = self.convert_window_index_to_original_time(300)
            return {
                'is_normal': False,
                'fault_class': self.results['fault_class'],
                'fault_time': self.results['fault_time'],
                'original_fault_time': self.results['original_fault_time']
            }
        is_normal = (fault_class == "normal")
        if is_normal:
            return {
                'is_normal': True,
                'fault_class': None,
                'fault_time': None,
                'original_fault_time': None
            }
        else:
            return {
                'is_normal': False,
                'fault_class': fault_class,
                'fault_time': self.results['fault_time'],
                'original_fault_time': self.results['original_fault_time']
            }

    def classify_normalized_data(self, normalized_data: np.ndarray) -> Dict[str, Any]:
        """
        정상화된 데이터를 입력받아 정상/비정상 판단 및 결과 반환 (Model4 역할)
        Returns:
            {
                'is_normal': bool,
                'fault_class': str or None,
                'fault_time': int or None,
                'original_fault_time': int or None
            }
        """
        return self.detect_fault(normalized_data)
    
    def get_results_for_model2(self) -> Dict[str, Any]:
        """
        Model2에게 전달할 결과 (슬라이딩 윈도우 인덱스 사용)
        
        Returns:
            Model2용 결과 딕셔너리
        """
        if self.results['fault_time'] is None:
            return None
        
        return {
            'fault_time': self.results['fault_time'],  # 슬라이딩 윈도우 인덱스 (0~4599)
            'fault_class': self.results['fault_class']
        }
    
    def get_results_for_llm(self) -> Dict[str, Any]:
        """
        LLM에게 전달할 결과 (원본 시점 사용)
        
        Returns:
            LLM용 결과 딕셔너리
        """
        if self.results['fault_time'] is None:
            return None
        
        return {
            'fault_time': self.results['original_fault_time'],  # 원본 시점 (0~959)
            'fault_class': self.results['fault_class']
        }
    
    def is_normal(self) -> bool:
        """
        정상 상태인지 확인합니다.
        
        Returns:
            정상이면 True, 비정상이면 False
        """
        return self.results['fault_class'] == "normal" 