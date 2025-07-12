"""
Tennessee Eastman Process 파이프라인

전체 파이프라인 구조:
1. Model1 (CNN1D2D + LSTM): Fault 시점 탐지 및 종류 분류
2. Model2 (Conditional TCN-AE): Fault 이후 조작변수 정상화
3. Model3 (RSSM): 정상화된 조작변수로 반응변수 예측
4. Model4 (Model1과 동일): 정상화 결과 검증
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import logging
from collections import defaultdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fault 타입 설명
FAULT_DESCRIPTIONS = {
    0: "정상 상태",
    1: "A/C 피드 비율, B 구성 일정 (공정 변수 11)",
    2: "B 구성 일정, A/C 비율 일정 (공정 변수 9)",
    3: "D 피드 온도 (공정 변수 7)",
    4: "반응기 냉각수 입구 온도 (공정 변수 8)",
    5: "응축기 냉각수 입구 온도 (공정 변수 11)",
    6: "A 피드 손실 (공정 변수 1)",
    7: "C 헤더 압력 손실 - 가용성 감소 (공정 변수 4)",
    8: "A, B, C 피드 구성 (공정 변수 4)",
    9: "D 피드 온도 (공정 변수 7)",
    10: "C 피드 온도 (공정 변수 3)",
    11: "반응기 냉각수 입구 온도 (공정 변수 8)",
    12: "응축기 냉각수 입구 온도 (공정 변수 11)"
}

@dataclass
class LLMData:
    """LLM에게 전달할 구조화된 데이터"""
    # Model1 결과
    fault_detection: Optional[Dict[str, Any]] = None
    # Model2 결과
    normalization: Optional[Dict[str, Any]] = None
    # Model3 결과
    prediction: Optional[Dict[str, Any]] = None
    # Model4 결과
    validation: Optional[Dict[str, Any]] = None

@dataclass
class PipelineData:
    """파이프라인 단계별 데이터를 저장하는 클래스"""
    # 입력 데이터
    input_data: np.ndarray
    timestamps: np.ndarray
    
    # Model1 결과
    fault_type: Optional[int] = None
    fault_onset_time: Optional[int] = None
    confidence: Optional[float] = None
    
    # Model2 결과
    normalized_manipulated_vars: Optional[np.ndarray] = None
    normalization_stats: Optional[Dict[str, Dict[str, float]]] = None
    
    # Model3 결과
    predicted_response_vars: Optional[np.ndarray] = None
    response_impact: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Model4 결과
    validation_result: Optional[Dict[str, Any]] = None
    
    # LLM 데이터
    llm_data: Optional[LLMData] = None
    
    def save(self, save_dir: str):
        """결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 각 단계별 결과 저장
        np.save(os.path.join(save_dir, 'input_data.npy'), self.input_data)
        np.save(os.path.join(save_dir, 'timestamps.npy'), self.timestamps)
        
        if self.llm_data:
            # LLM 데이터 저장
            import json
            with open(os.path.join(save_dir, 'llm_data.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'fault_detection': self.llm_data.fault_detection,
                    'normalization': self.llm_data.normalization,
                    'prediction': self.llm_data.prediction,
                    'validation': self.llm_data.validation
                }, f, ensure_ascii=False, indent=2)


class TEPPipeline:
    """Tennessee Eastman Process 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
                - model1_path: Model1 가중치 경로
                - model2_path: Model2 가중치 경로
                - model3_path: Model3 가중치 경로
                - device: 실행 디바이스 (cuda/cpu)
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # 모델 초기화 (실제 구현 시 각 모델 클래스로 교체)
        self.model1 = None  # Fault 분류기
        self.model2 = None  # 조작변수 정상화
        self.model3 = None  # 반응변수 예측
        self.model4 = None  # 검증 (Model1과 동일)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """모델 초기화 (실제 구현 시 각 모델 로드)"""
        logger.info("모델 초기화 중...")
        # TODO: 각 모델 구현 후 실제 초기화 코드 작성
        pass
    
    def _prepare_fault_detection_data(self, fault_type: int, onset_time: int,
                                    confidence: float) -> Dict[str, Any]:
        """Model1 결과를 LLM을 위한 구조화된 데이터로 변환"""
        return {
            "fault_type": fault_type,
            "fault_description": FAULT_DESCRIPTIONS.get(fault_type, "알 수 없는 Fault"),
            "onset_time": onset_time,
            "confidence": confidence
        }
    
    def _prepare_normalization_data(self, original_m: np.ndarray,
                                  normalized_m: np.ndarray,
                                  fault_onset: int) -> Dict[str, Any]:
        """Model2 결과를 LLM을 위한 구조화된 데이터로 변환"""
        # Fault 발생 이후 구간만 분석
        original_after = original_m[fault_onset:]
        normalized_after = normalized_m[fault_onset:]
        
        # 변수별 통계 계산
        var_changes = {}
        for i in range(original_m.shape[1]):
            orig_stats = {
                "mean": float(np.mean(original_after[:, i])),
                "std": float(np.std(original_after[:, i]))
            }
            norm_stats = {
                "mean": float(np.mean(normalized_after[:, i])),
                "std": float(np.std(normalized_after[:, i]))
            }
            percent_change = ((norm_stats["mean"] - orig_stats["mean"]) / 
                            orig_stats["mean"] * 100)
            
            var_changes[f"MV{i+1}"] = {
                "mean_before": orig_stats["mean"],
                "mean_after": norm_stats["mean"],
                "std_before": orig_stats["std"],
                "std_after": norm_stats["std"],
                "percent_change": float(percent_change)
            }
        
        # 변화가 큰 순서대로 정렬
        sorted_vars = sorted(
            var_changes.items(),
            key=lambda x: abs(x[1]["percent_change"]),
            reverse=True
        )
        top_changes = [var for var, _ in sorted_vars[:3]]
        
        return {
            "변수별_변화": var_changes,
            "주요_변화_순위": top_changes,
            "정상화_구간": {
                "start": fault_onset,
                "end": len(original_m)
            }
        }
    
    def _prepare_prediction_data(self, original_x: np.ndarray,
                               predicted_x: np.ndarray,
                               mv_changes: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Model3 결과를 LLM을 위한 구조화된 데이터로 변환"""
        # 반응변수별 변화 분석
        var_impacts = {}
        for i in range(original_x.shape[1]):
            current_mean = float(np.mean(original_x[:, i]))
            predicted_mean = float(np.mean(predicted_x[:, i]))
            percent_change = ((predicted_mean - current_mean) / current_mean * 100)
            
            # 이 반응변수에 영향을 준 주요 조작변수 찾기
            # (실제로는 더 복잡한 상관관계 분석이 필요)
            related_mvs = []
            for mv, stats in mv_changes.items():
                if abs(stats["percent_change"]) > 5:  # 임의의 임계값
                    related_mvs.append(mv)
            
            var_impacts[f"X{i+1}"] = {
                "expected_change": float(percent_change),
                "current_value": current_mean,
                "predicted_value": predicted_mean,
                "관련_조작변수": related_mvs[:3]  # 상위 3개만
            }
        
        # 변화가 큰 순서대로 정렬
        sorted_vars = sorted(
            var_impacts.items(),
            key=lambda x: abs(x[1]["expected_change"]),
            reverse=True
        )
        top_changes = [var for var, _ in sorted_vars[:3]]
        
        # 전체 시스템 영향도 계산 (0~1)
        # 여기서는 단순히 변화량의 평균을 정규화
        changes = [abs(v["expected_change"]) for v in var_impacts.values()]
        system_impact = min(1.0, sum(changes) / (len(changes) * 100))
        
        return {
            "영향받은_변수": var_impacts,
            "주요_변화_순위": top_changes,
            "전체_시스템_영향도": float(system_impact)
        }
    
    def _prepare_validation_data(self, final_fault_type: int,
                               confidence: float) -> Dict[str, Any]:
        """Model4 결과를 LLM을 위한 구조화된 데이터로 변환"""
        return {
            "final_state": "정상" if final_fault_type == 0 else "비정상",
            "fault_type": final_fault_type,
            "fault_description": FAULT_DESCRIPTIONS.get(final_fault_type, "알 수 없는 Fault"),
            "confidence": confidence
        }
    
    def run_model1(self, data: PipelineData) -> None:
        """
        Model1: Fault 시점 탐지 및 종류 분류
        
        Args:
            data: 입력 데이터를 포함한 PipelineData 객체
        """
        logger.info("Model1 실행: Fault 탐지 및 분류")
        # TODO: Model1 구현
        
        # 임시 더미 결과
        data.fault_type = 1
        data.fault_onset_time = 100
        data.confidence = 0.95
        
        # LLM 데이터 준비
        if data.llm_data is None:
            data.llm_data = LLMData()
        
        data.llm_data.fault_detection = self._prepare_fault_detection_data(
            data.fault_type,
            data.fault_onset_time,
            data.confidence
        )
    
    def run_model2(self, data: PipelineData) -> None:
        """
        Model2: Fault 이후 조작변수 정상화
        
        Args:
            data: Model1 결과를 포함한 PipelineData 객체
        """
        logger.info("Model2 실행: 조작변수 정상화")
        # TODO: Model2 구현
        
        # 임시 더미 결과
        data.normalized_manipulated_vars = data.input_data.copy()
        
        # LLM 데이터 준비
        if data.llm_data is None:
            data.llm_data = LLMData()
        
        data.llm_data.normalization = self._prepare_normalization_data(
            data.input_data,
            data.normalized_manipulated_vars,
            data.fault_onset_time
        )
    
    def run_model3(self, data: PipelineData) -> None:
        """
        Model3: 정상화된 조작변수로 반응변수 예측
        
        Args:
            data: Model2 결과를 포함한 PipelineData 객체
        """
        logger.info("Model3 실행: 반응변수 예측")
        # TODO: Model3 구현
        
        # 임시 더미 결과
        data.predicted_response_vars = data.input_data.copy()
        
        # LLM 데이터 준비
        if data.llm_data is None:
            data.llm_data = LLMData()
        
        data.llm_data.prediction = self._prepare_prediction_data(
            data.input_data,
            data.predicted_response_vars,
            data.llm_data.normalization["변수별_변화"]
        )
    
    def run_model4(self, data: PipelineData) -> None:
        """
        Model4: 정상화 결과 검증
        
        Args:
            data: Model3 결과를 포함한 PipelineData 객체
        """
        logger.info("Model4 실행: 결과 검증")
        # TODO: Model4 구현 (Model1과 동일한 구조 사용)
        
        # 임시 더미 결과
        final_fault_type = 0  # 정상화 성공
        confidence = 0.90
        
        data.validation_result = {
            'final_fault_type': final_fault_type,
            'confidence': confidence
        }
        
        # LLM 데이터 준비
        if data.llm_data is None:
            data.llm_data = LLMData()
        
        data.llm_data.validation = self._prepare_validation_data(
            final_fault_type,
            confidence
        )
    
    def run_pipeline(self, input_data: np.ndarray, timestamps: np.ndarray,
                    save_dir: Optional[str] = None) -> PipelineData:
        """
        전체 파이프라인 실행
        
        Args:
            input_data: 입력 데이터
            timestamps: 시계열 데이터의 타임스탬프
            save_dir: 결과 저장 디렉토리 (옵션)
            
        Returns:
            PipelineData: 모든 단계의 결과를 포함한 객체
        """
        logger.info("파이프라인 실행 시작")
        
        # 데이터 객체 초기화
        data = PipelineData(
            input_data=input_data,
            timestamps=timestamps
        )
        
        # 각 모델 순차적 실행
        self.run_model1(data)
        self.run_model2(data)
        self.run_model3(data)
        self.run_model4(data)
        
        # 결과 저장
        if save_dir:
            data.save(save_dir)
            logger.info(f"파이프라인 결과 저장 완료: {save_dir}")
        
        logger.info("파이프라인 실행 완료")
        return data


def create_pipeline(config: Dict[str, Any]) -> TEPPipeline:
    """
    파이프라인 생성 헬퍼 함수
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        TEPPipeline: 초기화된 파이프라인 객체
    """
    return TEPPipeline(config) 