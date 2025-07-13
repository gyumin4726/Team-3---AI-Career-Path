"""
Tennessee Eastman Process (TEP) 전체 파이프라인
4단계 공정 이상 분석 및 정상화 시스템
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import sys
import os
from collections import Counter

# 상위 디렉토리 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LLM'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from LLM import LLM
from model1_module import Model1Module
from model3_module import Model3Module
from src.data.dataset import TEPNPYDataset, CSVToTensor

class TEPPipeline:
    """
    TEP 전체 파이프라인 관리 클래스
    
    4단계 파이프라인:
    1. Model1: Fault 시점 탐지 + Fault 종류 분류
    2. Model2: 조작 변수 정상화 (Conditional TCN-AE)
    3. Model3: 반응 변수 예측 (RSSM)
    4. Model4: 정상 여부 재분류 (Model1 재사용)
    """
    
    def __init__(self):
        """파이프라인 초기화"""
        self.llm = LLM()
        self.normalized_m = None
        self.predicted_x = None
        self.final_class = None
        
        # Model1 모듈 초기화
        self.model1_module = Model1Module()
        
        # Model3 모듈 초기화
        self.model3_module = Model3Module()
        
    def run_full_pipeline(self, data_sequence: np.ndarray) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Args:
            data_sequence: 전체 데이터 시퀀스 (B, 50, 52)
            
        Returns:
            결과 딕셔너리
        """
        print("TEP 전체 파이프라인 시작")
        print("="*60)
        print(f"입력 데이터 형태: {data_sequence.shape}")
        
        # 1단계: Fault 시점 탐지 + Fault 종류 분류 (Model1)
        model1_results = self.model1_module.detect_fault(data_sequence)
        fault_time = model1_results.get('fault_time')
        fault_class = model1_results.get('fault_class')
        
        # Model1 결과에 따른 분기 처리
        if self.model1_module.is_normal():
            print("정상 상태 감지 - 파이프라인 종료")
            
            # LLM에게 Model1 결과 전달
            model1_results = self.get_model1_results_for_llm()
            model1_explanation = self.llm.generate_response(
                f"Model1 결과를 설명해주세요: {model1_results}"
            )
            
            results = {
                'fault_time': fault_time,
                'fault_class': fault_class,
                'normalized_m': None,
                'predicted_x': None,
                'final_class': 'normal',
                'success': True,
                'pipeline_status': 'early_termination_normal',
                'llm_explanations': {
                    'model1': model1_explanation
                }
            }
            print("="*60)
            print("파이프라인 완료 (정상 상태)")
            return results
        
        # 비정상인 경우: 반복 정상화 시도
        max_iterations = 3  # 최대 반복 횟수
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
            print(f"비정상 상태 감지: {fault_class} - 반복 {current_iteration}/{max_iterations}")
            
            # TODO: 나중에 m과 x 분리 로직 구현
            # 현재는 전체 데이터를 m과 x로 동일하게 사용
            m_sequence = data_sequence  # (B, 50, 52) - 나중에 조작 변수만 추출
            x_sequence = data_sequence  # (B, 50, 52) - 나중에 반응 변수만 추출
            
            # Model2용 결과 가져오기 (슬라이딩 윈도우 인덱스)
            model2_results = self.model1_module.get_results_for_model2()
            fault_time_for_model2 = model2_results['fault_time']  # 0~4599 범위
            
            # 2단계: 조작 변수 정상화 (Model2에 fault 정보 전달)
            # TODO: Model2 구현 필요
            normalized_m = m_sequence.copy()  # 임시로 원본 복사
            
            # 3단계: 반응 변수 예측 (Model3에 fault 정보 전달)
            predicted_x = self.model3_module.predict_new_sequence(x_sequence, fault_time_for_model2)
            
            # 정상화된 데이터 결합
            # Model3이 이미 전체 데이터를 반환하므로 그대로 사용
            normalized_data = predicted_x
            
            print(f"정상화된 데이터 결합 완료: {normalized_data.shape}")
            
            # 4단계: 정상 여부 재분류 (Model4 = Model1 재사용)
            model4_results = self.model1_module.detect_fault(normalized_data)
            fault_time = model4_results.get('fault_time')
            final_class = model4_results.get('fault_class')
            
            # Model4 결과에 따른 분기 처리
            print(f"DEBUG: final_class = '{final_class}', type = {type(final_class)}")
            if final_class == "normal":
                print(f"반복 {current_iteration}: 정상화 완료!")
                
                # LLM에게 Model1과 Model3 결과 전달
                model1_results = self.get_model1_results_for_llm()
                model1_explanation = self.llm.generate_response(
                    f"Model1 결과를 설명해주세요: {model1_results}"
                )
                
                model3_results = self.get_model3_results_for_llm(data_sequence, fault_time_for_model2)
                model3_explanation = self.llm.generate_response(
                    f"Model3 결과를 설명해주세요: {model3_results}"
                )
                
                results = {
                    'fault_time': fault_time,  # LLM용 원본 시점 (0~959)
                    'fault_class': fault_class,
                    'normalized_m': normalized_m,
                    'predicted_x': predicted_x,
                    'final_class': final_class,
                    'success': True,
                    'pipeline_status': f'normalized_after_iteration_{current_iteration}',
                    'iterations': current_iteration,
                    'llm_explanations': {
                        'model1': model1_explanation,
                        'model3': model3_explanation
                    }
                }
                print("="*60)
                print(f"파이프라인 완료! (반복 {current_iteration}회 후 정상화 성공)")
                return results
            else:
                print(f"반복 {current_iteration}: 정상화 실패, 다시 시도...")
                # 다음 반복을 위해 현재 정상화된 데이터를 새로운 입력으로 사용
                data_sequence = normalized_data
                fault_class = final_class
        
        # 최대 반복 횟수 초과
        print(f"최대 반복 횟수({max_iterations}) 초과 - 정상화 실패")
        
        # LLM에게 Model1과 Model3 결과 전달
        model1_results = self.get_model1_results_for_llm()
        model1_explanation = self.llm.generate_response(
            f"Model1 결과를 설명해주세요: {model1_results}"
        )
        
        model3_results = self.get_model3_results_for_llm(data_sequence, fault_time_for_model2)
        model3_explanation = self.llm.generate_response(
            f"Model3 결과를 설명해주세요: {model3_results}"
        )
        
        results = {
            'fault_time': fault_time,  # LLM용 원본 시점 (0~959)
            'fault_class': fault_class,
            'normalized_m': normalized_m,
            'predicted_x': predicted_x,
            'final_class': final_class,
            'success': False,
            'pipeline_status': 'max_iterations_exceeded',
            'iterations': max_iterations,
            'llm_explanations': {
                'model1': model1_explanation,
                'model3': model3_explanation
            }
        }
        
        print("="*60)
        print("파이프라인 완료! (정상화 실패)")
        return results
    
    def get_model1_results_for_llm(self) -> Dict[str, Any]:
        """
        Model1의 결과를 LLM에게 전달하기 위한 형태로 반환합니다.
        
        Returns:
            Model1 결과 딕셔너리
        """
        return self.model1_module.get_results_for_llm()
    
    def get_model3_results(self) -> Dict[str, Any]:
        """
        Model3의 결과를 반환합니다.
        
        Returns:
            Model3 결과 딕셔너리
        """
        return self.model3_module.get_results()

    def get_model3_results_for_llm(self, original_sequence: np.ndarray, fault_time: int) -> Dict[str, Any]:
        """
        Model3의 결과를 LLM에게 전달하기 위한 형태로 반환합니다.
        
        Args:
            original_sequence: (B, 50, 52) - 원본 입력 데이터
            fault_time: 슬라이딩 윈도우 인덱스 기준 fault 시점
            
        Returns:
            Model3 LLM용 결과 딕셔너리
        """
        return self.model3_module.get_results_for_llm(original_sequence, fault_time)

def main():
    """메인 실행 함수"""
    print("TEP 파이프라인 테스트 시작")
    
    # 실제 TEP 데이터 로드
    try:
        # 데이터 변환 설정
        transform = CSVToTensor()
        
        # 테스트 데이터셋 생성
        test_dataset = TEPNPYDataset(
            data_path='data/final_X.npy',
            labels_path='data/final_Y.npy',
            transform=transform,
            is_test=True
        )
        
        print(f"데이터셋 로드 완료: {len(test_dataset)}개 샘플")
        
        # 첫 번째 배치 데이터 추출 (92개 샘플 - 한 시뮬레이션 전체)
        batch_size = 92
        batch_data = []
        batch_labels = []
        
        for i in range(min(batch_size, len(test_dataset))):
            sample = test_dataset[i]
            batch_data.append(sample['shot'].numpy())  # (50, 52)
            batch_labels.append(sample['label'].item())
        
        # 배치 데이터를 numpy 배열로 변환
        batch_data_array = np.array(batch_data)  # (92, 50, 52) - 한 시뮬레이션 전체
        
        print(f"테스트 데이터 준비:")
        print(f"  - batch_data_array: {batch_data_array.shape}")
        print(f"  - 총 시점 수: {batch_data_array.shape[0] * batch_data_array.shape[1]} = {batch_data_array.shape[0] * batch_data_array.shape[1]}개")
        
        # 파이프라인 실행 (전체 데이터 전달)
        pipeline = TEPPipeline()
        results = pipeline.run_full_pipeline(batch_data_array)
    
    except FileNotFoundError as e:
        print(f"데이터 파일을 찾을 수 없습니다: {e}")
        print("랜덤 데이터로 테스트를 진행합니다.")
        
        # 랜덤 데이터 생성 (fallback)
        np.random.seed(42)
        batch_size = 92
        batch_data_array = np.random.randn(batch_size, 50, 52)  # (92, 50, 52) - 전체 시퀀스
        
        print(f"랜덤 테스트 데이터 생성:")
        print(f"  - batch_data_array: {batch_data_array.shape}")
        
        # 파이프라인 실행 (랜덤 데이터 전달)
        pipeline = TEPPipeline()
        results = pipeline.run_full_pipeline(batch_data_array)
    
    # LLM으로 결과 설명
    print("\n" + "="*60)
    print("LLM 결과 해설:")
    if 'llm_explanations' in results:
        print("Model1 설명:", results['llm_explanations'].get('model1', 'N/A'))
        if 'model3' in results['llm_explanations']:
            print("Model3 설명:", results['llm_explanations']['model3'])
    else:
        print("LLM 설명이 생성되지 않았습니다.")

if __name__ == "__main__":
    main() 