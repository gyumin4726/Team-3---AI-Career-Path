import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import TEPNPYDataset, CSVToTensor
from torch.utils.data import DataLoader
import click
import logging
from pathlib import Path
import os
from src.models.convolutional_models import CNN1D2DDiscriminatorMultitask

class WrapperModel(torch.nn.Module):
    """SHAP 분석을 위한 모델 래퍼"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        type_logits, _ = self.model(x)
        return type_logits  # SHAP은 단일 Tensor만 기대

def load_model(model_path, device, input_size, class_count):
    """체크포인트에서 모델 로드 - 올바른 방식으로 구현"""
    model = CNN1D2DDiscriminatorMultitask(
        input_size=input_size,
        n_layers_1d=4,
        n_layers_2d=4,
        n_channel=input_size * 3,
        n_channel_2d=100,
        class_count=class_count,
        kernel_size=9,
        dropout=0.2,
        groups=input_size
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['discriminator_state_dict'])
    model.eval()
    return WrapperModel(model)  # 래퍼 모델 반환

def prepare_background_data(n_samples=100):
    """SHAP 분석을 위한 배경 데이터 준비"""
    transform = CSVToTensor()
    dataset = TEPNPYDataset(
        data_path="data/train_X.npy",
        labels_path="data/train_intY.npy",
        transform=transform,
        is_test=False
    )
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    background_data = next(iter(loader))
    return background_data, dataset.features_count, dataset.class_count

def create_shap_plots(shap_values, test_data, explainer, output_dir, class_idx=0):
    """SHAP 값 시각화 - 클래스별 분석"""
    # 전체 특성에 대한 요약 플롯
    plt.figure(figsize=(15, 10))
    shap.summary_plot(
        shap_values[class_idx],  # class_idx에 해당하는 클래스의 SHAP 값만 사용
        test_data["shot"].cpu().numpy(),
        show=False
    )
    plt.title(f'SHAP Summary Plot for Class {class_idx}')
    plt.savefig(os.path.join(output_dir, f'shap_summary_class_{class_idx}.png'))
    plt.close()

    # 첫 번째 샘플에 대한 워터폴 플롯
    sample_idx = 0
    shap_value = shap.Explanation(
        values=shap_values[class_idx][sample_idx],
        base_values=explainer.expected_value[class_idx],
        data=test_data["shot"][sample_idx].cpu().numpy()
    )
    plt.figure(figsize=(20, 10))
    shap.plots.waterfall(shap_value, show=False)
    plt.title(f'SHAP Waterfall Plot for Class {class_idx}, Sample {sample_idx}')
    plt.savefig(os.path.join(output_dir, f'shap_waterfall_class_{class_idx}_sample_{sample_idx}.png'))
    plt.close()

@click.command()
@click.option('--model_path', required=True, type=str, help='분석할 모델의 체크포인트 경로')
@click.option('--output_dir', required=True, type=str, help='결과 저장 디렉토리')
@click.option('--n_samples', default=100, type=int, help='분석할 샘플 수')
@click.option('--target_class', default=0, type=int, help='분석할 타겟 클래스 (fault type)')
def main(model_path, output_dir, n_samples, target_class):
    """SHAP 분석 실행"""
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 배경 데이터 준비 및 모델 파라미터 얻기
    background_data, input_size, class_count = prepare_background_data(n_samples)
    logger.info(f"Prepared background data: {background_data['shot'].shape}")
    logger.info(f"Input size: {input_size}, Class count: {class_count}")

    # 모델 로드 - 올바른 방식으로
    model = load_model(model_path, device, input_size, class_count)
    logger.info("Model loaded successfully")

    # SHAP 설명자 생성
    explainer = shap.DeepExplainer(model, background_data["shot"].to(device))
    logger.info("Created SHAP explainer")

    # 분석할 데이터 준비
    test_data, _, _ = prepare_background_data(10)  # 10개 샘플만 분석
    
    # SHAP 값 계산
    shap_values = explainer.shap_values(test_data["shot"].to(device))
    logger.info("Calculated SHAP values")

    # 결과 시각화 및 저장
    create_shap_plots(shap_values, test_data, explainer, output_dir, target_class)
    logger.info(f"Results saved to {output_dir}")

    # 추가: 모든 클래스에 대한 분석 (선택적)
    for class_idx in range(class_count):
        if class_idx != target_class:
            create_shap_plots(shap_values, test_data, explainer, output_dir, class_idx)
            logger.info(f"Generated plots for class {class_idx}")

if __name__ == '__main__':
    main() 