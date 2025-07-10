import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

def analyze_data_structure(file_path):
    """데이터의 구조를 분석합니다."""
    # fault 1-12만 로드
    df = pd.read_csv(file_path)
    df = df[df['faultNumber'] <= 12]
    
    print("\n=== 데이터 기본 정보 ===")
    print("\n전체 데이터 크기:", df.shape)
    print("\nFault 번호별 데이터 수:")
    print(df['faultNumber'].value_counts().sort_index())
    print("\n각 Fault별 Simulation 수:")
    print(df.groupby('faultNumber')['simulationRun'].nunique())
    return df

def load_tep_data(df):
    """데이터프레임에서 측정변수(XMEAS)와 조작변수(XMV)를 분리합니다."""
    # 실제 컬럼명 분리
    xmeas_cols = [col for col in df.columns if col.startswith('xmeas_')]
    xmv_cols = [col for col in df.columns if col.startswith('xmv_')]
    
    print("\n측정변수(XMEAS) 개수:", len(xmeas_cols))
    print("조작변수(XMV) 개수:", len(xmv_cols))
    
    XMEAS = df[xmeas_cols]
    XMV = df[xmv_cols]
    
    return XMEAS, XMV

def calculate_correlations_for_simulation(XMEAS, XMV):
    """하나의 시뮬레이션에 대한 상관계수를 계산합니다."""
    # 모든 변수를 하나의 데이터프레임으로 합치고 상관계수 계산
    combined = pd.concat([XMEAS, XMV], axis=1)
    corr_matrix = combined.corr()
    
    # XMEAS와 XMV 사이의 상관계수만 추출
    return corr_matrix.loc[XMEAS.columns, XMV.columns]

def calculate_average_correlations(df, fault_number):
    """특정 fault에 대해 모든 simulation의 평균 상관계수를 계산합니다."""
    fault_data = df[df['faultNumber'] == fault_number]
    
    # 변수 컬럼 추출
    xmeas_cols = [col for col in df.columns if col.startswith('xmeas_')]
    xmv_cols = [col for col in df.columns if col.startswith('xmv_')]
    
    # 각 simulation별 상관계수 계산
    correlation_matrices = []
    for sim_num in tqdm(fault_data['simulationRun'].unique(), 
                       desc=f'Fault {fault_number} - Processing simulations',
                       leave=False):
        sim_data = fault_data[fault_data['simulationRun'] == sim_num]
        XMEAS = sim_data[xmeas_cols]
        XMV = sim_data[xmv_cols]
        corr_matrix = calculate_correlations_for_simulation(XMEAS, XMV)
        correlation_matrices.append(corr_matrix)
    
    # 평균 상관계수 계산
    avg_correlation = sum(correlation_matrices) / len(correlation_matrices)
    return avg_correlation

def plot_correlation_heatmap(corr_matrix, fault_number, output_path=None):
    """상관계수 행렬을 히트맵으로 시각화합니다."""
    if output_path is None:
        output_path = f'correlation_heatmap_fault{fault_number}.png'
        
    plt.figure(figsize=(15, 20))
    sns.heatmap(corr_matrix, 
                cmap='RdBu_r',
                center=0,
                vmin=-1, 
                vmax=1,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(f'Average Correlation between XMEAS and XMV\nFault {fault_number} (Averaged over {500} simulations)')
    plt.ylabel('Process Variables (XMEAS)')
    plt.xlabel('Manipulated Variables (XMV)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def find_strongest_correlations(corr_matrix, threshold=0.3):
    """각 측정변수에 대해 가장 강한 상관관계를 가진 조작변수들을 찾습니다."""
    strong_correlations = []
    
    for xmeas in corr_matrix.index:
        correlations = corr_matrix.loc[xmeas]
        strong_xmvs = correlations[abs(correlations) >= threshold]
        
        if not strong_xmvs.empty:
            strong_correlations.append({
                'XMEAS': xmeas,
                'Strong correlations with XMV': ', '.join([f"{col} ({correlations[col]:.2f})" 
                                       for col in strong_xmvs.index])
            })
    
    return pd.DataFrame(strong_correlations)

def compare_fault_patterns(all_correlations):
    """각 fault별 상관관계 패턴을 비교 분석합니다."""
    analysis_text = []
    analysis_text.append("=== TEP 공정의 Fault별 상관관계 패턴 분석 ===\n")
    
    # 1. 각 fault별 특징적인 상관관계 패턴 분석
    for fault_num, corr_matrix in all_correlations.items():
        # 가장 강한 상관관계 찾기
        strongest_corr = np.abs(corr_matrix).max().max()
        strongest_pair = np.where(np.abs(corr_matrix) == strongest_corr)
        xmeas_idx, xmv_idx = strongest_pair[0][0], strongest_pair[1][0]
        
        analysis_text.append(f"\n[Fault {fault_num} 분석]")
        analysis_text.append(f"1. 가장 강한 상관관계: {corr_matrix.index[xmeas_idx]} - {corr_matrix.columns[xmv_idx]} ({corr_matrix.iloc[xmeas_idx, xmv_idx]:.3f})")
        
        # 전반적인 상관관계 강도
        avg_corr = np.abs(corr_matrix).mean().mean()
        analysis_text.append(f"2. 평균 상관관계 강도: {avg_corr:.3f}")
        
        # 주요 영향을 받는 측정변수들
        affected_vars = corr_matrix.index[np.any(np.abs(corr_matrix) > 0.5, axis=1)]
        analysis_text.append(f"3. 강한 영향을 받는 측정변수들 (|상관계수| > 0.5):")
        for var in affected_vars:
            strong_xmvs = corr_matrix.loc[var][np.abs(corr_matrix.loc[var]) > 0.5]
            correlations = [f"{col} ({val:.2f})" for col, val in strong_xmvs.items()]
            analysis_text.append(f"   - {var}: {', '.join(correlations)}")
    
    # 2. Fault 간 패턴 유사성 분석
    analysis_text.append("\n\n=== Fault 간 패턴 유사성 분석 ===")
    
    # 상관행렬을 1차원 벡터로 변환하여 비교
    fault_vectors = {f: corr_matrix.values.flatten() for f, corr_matrix in all_correlations.items()}
    similarity_matrix = np.zeros((12, 12))
    
    for i in range(1, 13):
        for j in range(1, 13):
            if i <= j:
                similarity = np.corrcoef(fault_vectors[i], fault_vectors[j])[0, 1]
                similarity_matrix[i-1, j-1] = similarity
                similarity_matrix[j-1, i-1] = similarity
    
    # 유사한 fault 그룹 찾기
    for i in range(12):
        similar_faults = []
        for j in range(12):
            if i != j and similarity_matrix[i, j] > 0.7:  # 0.7은 유사성 임계값
                similar_faults.append(f"Fault {j+1} ({similarity_matrix[i, j]:.2f})")
        
        if similar_faults:
            analysis_text.append(f"\nFault {i+1}과 유사한 패턴을 보이는 Fault들:")
            analysis_text.append(", ".join(similar_faults))
    
    # 3. 종합 분석
    analysis_text.append("\n\n=== 종합 분석 ===")
    
    # 전반적인 패턴 분석
    all_corrs = np.array([matrix.values for matrix in all_correlations.values()])
    global_strongest = np.abs(all_corrs).max()
    global_mean = np.abs(all_corrs).mean()
    
    analysis_text.append(f"1. 전체 Fault에서 가장 강한 상관관계: {global_strongest:.3f}")
    analysis_text.append(f"2. 전체 평균 상관관계 강도: {global_mean:.3f}")
    
    # 파일로 저장
    with open('fault_correlation_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis_text))

def main():
    # 데이터 로드 및 구조 분석
    df = analyze_data_structure('TEP_Faulty_Training.csv')
    
    # 각 fault별 상관관계 저장
    all_correlations = {}
    
    # 각 fault별로 분석
    for fault_number in tqdm(range(1, 13), desc='Processing faults'):
        print(f"\n\n{'='*50}")
        print(f"=== Fault {fault_number} 분석 ===")
        print('='*50)
        
        # 평균 상관계수 계산
        avg_corr_matrix = calculate_average_correlations(df, fault_number)
        all_correlations[fault_number] = avg_corr_matrix
        
        # 히트맵 생성
        plot_correlation_heatmap(avg_corr_matrix, fault_number)
        print(f"히트맵이 'correlation_heatmap_fault{fault_number}.png' 파일로 저장되었습니다.")
        
        # 강한 상관관계 분석
        print(f"\n=== Fault {fault_number}의 강한 상관관계 ===")
        strong_corr = find_strongest_correlations(avg_corr_matrix)
        print(strong_corr.to_string(index=False))
        print("\n")
    
    # fault 간 패턴 비교 분석
    print("\n모든 Fault의 상관관계 패턴 비교 분석 중...")
    compare_fault_patterns(all_correlations)
    print("분석 결과가 'fault_correlation_analysis.txt' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main() 