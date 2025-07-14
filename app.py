import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from typing import Dict, Any, Optional
import time
import gdown

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'main'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'LLM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'model1'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data'))

# TEP 파이프라인 임포트
try:
    from src.main import TEPPipeline
    from src.data import TEPNPYDataset
except ImportError as e:
    st.error(f"모듈 임포트 오류: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="Tennessee Eastman Process - 공정 이상 탐지 및 정상화",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pipeline-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-step {
        border-left-color: #28a745;
    }
    .warning-step {
        border-left-color: #ffc107;
    }
    .error-step {
        border-left-color: #dc3545;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """샘플 데이터 로드"""
    try:
        # 샘플 데이터 생성 (실제 데이터가 없는 경우)
        np.random.seed(30)
        # 파이프라인이 기대하는 형태: (92, 50, 52) - 한 시뮬레이션 전체
        sample_data = np.random.randn(92, 50, 52)  # 92개 배치, 50 시점, 52 센서
        return sample_data
    except Exception as e:
        st.error(f"샘플 데이터 로드 실패: {e}")
        return None

def create_sensor_plot(data: np.ndarray, title: str = "센서 데이터 시각화"):
    """센서 데이터 플롯 생성"""
    if data is None or data.size == 0:
        return None
    
    # 데이터 형태 변환: (B, T, S) → (T, S)
    if len(data.shape) == 3:
        data_2d = data.mean(axis=0)  # 배치 평균
    else:
        data_2d = data
    
    # 센서별로 색상 구분
    colors = px.colors.qualitative.Set3[:data_2d.shape[1]]
    
    fig = go.Figure()
    
    for i in range(min(10, data_2d.shape[1])):  # 처음 10개 센서만 표시
        fig.add_trace(go.Scatter(
            y=data_2d[:, i],
            mode='lines',
            name=f'센서 {i+1}',
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="시점",
        yaxis_title="센서 값",
        height=400,
        showlegend=True
    )
    
    return fig

def create_pipeline_flowchart():
    """파이프라인 플로우차트 생성"""
    fig = go.Figure()
    
    # 노드 정의
    nodes = [
        {'id': 'input', 'x': 0, 'y': 0, 'label': '입력 데이터\n(52개 센서)'},
        {'id': 'model1', 'x': 2, 'y': 0, 'label': 'Model1\nFault 탐지 + 분류'},
        {'id': 'normal', 'x': 4, 'y': 1, 'label': '정상 상태\n→ 종료'},
        {'id': 'model2', 'x': 2, 'y': -1, 'label': 'Model2\n조작 변수 정상화'},
        {'id': 'model3', 'x': 4, 'y': -1, 'label': 'Model3\n반응 변수 예측'},
        {'id': 'model4', 'x': 6, 'y': -1, 'label': 'Model4\n정상 여부 재분류'},
        {'id': 'success', 'x': 8, 'y': 0, 'label': '정상화 완료'},
        {'id': 'retry', 'x': 6, 'y': -2, 'label': '재시도\n(최대 3회)'}
    ]
    
    # 엣지 정의
    edges = [
        ('input', 'model1'),
        ('model1', 'normal'),
        ('model1', 'model2'),
        ('model2', 'model3'),
        ('model3', 'model4'),
        ('model4', 'success'),
        ('model4', 'retry'),
        ('retry', 'model2')
    ]
    
    # 노드 그리기
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']], y=[node['y']],
            mode='markers+text',
            marker=dict(size=50, color='lightblue'),
            text=node['label'].split('\n'),
            textposition="middle center",
            showlegend=False,
            hoverinfo='text'
        ))
    
    # 엣지 그리기
    for edge in edges:
        start_node = next(n for n in nodes if n['id'] == edge[0])
        end_node = next(n for n in nodes if n['id'] == edge[1])
        
        fig.add_trace(go.Scatter(
            x=[start_node['x'], end_node['x']],
            y=[start_node['y'], end_node['y']],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="TEP 4단계 파이프라인 플로우",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        showlegend=False
    )
    
    return fig

def run_tep_pipeline(data: np.ndarray):
    """TEP 파이프라인 실행"""
    try:
        # 파이프라인 초기화
        pipeline = TEPPipeline()
        
        # 파이프라인 실행
        results = pipeline.run_full_pipeline(data)
        
        return results
    except Exception as e:
        st.error(f"파이프라인 실행 오류: {e}")
        return None

def display_results(results: Dict[str, Any]):
    """결과 표시"""
    if results is None:
        return
    
    st.subheader("📊 분석 결과")
    
    # 기본 정보
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("초기 결함 유형", results.get('model1_fault_class', 'N/A'))
    
    with col2:
        fault_time = results.get('model1_fault_time', None)
        if fault_time is not None:
            total_minutes = fault_time * 3
            hours = total_minutes // 60
            minutes = total_minutes % 60
            st.metric("결함 시점", f"{total_minutes}분, {hours}시간 {minutes}분")
        else:
            st.metric("결함 시점", "없음")
    
    with col3:
        success = results.get('success', False)
        status = "✅ 성공" if success else "❌ 실패"
        st.metric("정상화 상태", status)
    
    # 파이프라인 상태
    pipeline_status = results.get('pipeline_status', 'unknown')
    if pipeline_status == 'early_termination_normal':
        st.success("🎉 정상 상태로 감지되어 파이프라인이 조기 종료되었습니다.")
    elif 'normalized' in pipeline_status:
        iterations = results.get('iterations', 0)
        st.success(f"🎉 {iterations}회 반복 후 정상화에 성공했습니다!")
    else:
        st.warning("⚠️ 최대 반복 횟수 초과로 정상화에 실패했습니다.")
    
    # LLM 설명
    if 'llm_explanations' in results:
        st.subheader("🤖 AI 분석 설명")
        
        explanations = results['llm_explanations']
        
        if 'model1' in explanations:
            with st.expander("Model1 (Fault 탐지 + 분류) 분석"):
                st.write(explanations['model1'])
        
        if 'model2' in explanations:
            with st.expander("Model2 (조작 변수 정상화) 분석"):
                st.write(explanations['model2'])
        
        if 'model3' in explanations:
            with st.expander("Model3 (반응 변수 예측) 분석"):
                st.write(explanations['model3'])

def download_large_file():
    url = "https://drive.google.com/uc?id=1AsYglG0Jm2Yi316LfjW7P7ThlBzxmghJ"
    output_path = "data/normal_db.npy"
    if not os.path.exists(output_path):
        os.makedirs("data", exist_ok=True)
        gdown.download(url, output_path, quiet=False)
        print("✅ Downloaded normal_db.npy")
    else:
        print("✅ File already exists")

# 다운로드 수행
# Streamlit 앱 실행 시 항상 먼저 체크

def main():
    download_large_file()
    # 메인 헤더
    st.markdown('<h1 class="main-header">🏭 Tennessee Eastman Process</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">공정 이상 탐지 및 정상화 파이프라인</h2>', unsafe_allow_html=True)
    
    # 메인 컨텐츠
    tab1, tab2, tab3 = st.tabs(["📋 프로젝트 개요", "🔬 파이프라인 분석", "📈 결과 시각화"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('''
            <div style="background: #f8f9fa; border-radius: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 36px 20px; min-width: 220px; text-align: center;">
                <h2 style="font-size: 1.3rem; font-weight: 700; color: #444; margin-bottom: 1rem;">팀 정보</h2>
                <div style="font-size: 1.08rem; line-height: 2; color: #333; font-weight: 500;">
                    2025 AI 커리어패스 프로그램 - 팀3<br/>
                    국민대학교 <b>박규민</b><br/>
                    동양미래대학교 <b>방석영</b><br/>
                    세종대학교 <b>엄태호</b><br/>
                    한양여자대학교 <b>조유영</b>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown('''
            <div style="background: #f8f9fa; border-radius: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 36px 20px; min-width: 240px; max-width: 340px; margin: 0 auto; text-align: center;">
                <h2 style="font-size: 1.3rem; font-weight: 700; color: #444; margin-bottom: 1rem;">데이터셋 정보</h2>
                <ul style="font-size: 1.05rem; line-height: 2; color: #333; font-weight: 500; text-align: left; margin-left: 1.2em;">
                    <li><b>센서 개수:</b> 52개 (22개 공정, 19개 분석, 11개 조작)</li>
                    <li><b>결함 유형:</b> 21가지 (정상 포함)</li>
                    <li><b>샘플링 주기:</b> 3분</li>
                    <li><b>슬라이딩 윈도우:</b> 960시점 → 윈도우 50, 스텝 10, 총 4600개</li>
                </ul>
                <div style="margin-top: 1.1rem; font-size: 0.98rem; color: #555;">
                    <b>데이터 구조 예시</b><br/>
                    <code>(B, 50, 52)</code><br/>
                    <span style="font-size:0.97rem;">B: 배치(런 수), 50: 윈도우, 52: 센서</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown('''
            <div style="background: #f8f9fa; border-radius: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); padding: 36px 20px; min-width: 240px; max-width: 360px; margin: 0 auto; text-align: center;">
                <h2 style="font-size: 1.3rem; font-weight: 700; color: #444; margin-bottom: 1rem;">파이프라인 원리</h2>
                <ol style="font-size: 1.05rem; line-height: 2; color: #333; font-weight: 500; text-align: left; margin-left: 1.2em;">
                    <li><b>Model1:</b> Fault 탐지 + 분류<br/>CNN1D2D로 고장 시점/종류 탐지, 정상시 종료</li>
                    <li><b>Model2:</b> 조작 변수 정상화<br/>KNN 기반 정상 DB와 비교, 고장 이후만 보정</li>
                    <li><b>Model3:</b> 반응 변수 예측<br/>TCNSeq2Seq로 정상화된 조작 변수 기반 예측</li>
                    <li><b>Model4:</b> 정상 여부 재분류<br/>Model1 재사용, 정상화 성공시 종료, 아니면 반복(최대 3회)</li>
                </ol>
                <div style="margin-top: 1.1rem; font-size: 0.98rem; color: #555;">
                    <b>반복 정상화 파이프라인</b><br/>
                    비정상 상태가 계속되면 최대 3회 반복<br/>
                    LLM 기반 결과 해설 제공
                </div>
            </div>
            ''', unsafe_allow_html=True)

    with tab2:
        st.subheader("파이프라인 분석")
        st.subheader("📊 데이터 로드")
        # test_per_fault 폴더의 정상, 0~12 fault 선택지 제공
        fault_options = [
            ("정상", "test_per_fault/fault_00_X.npy")
        ] + [
            (f"Fault {i:02d}", f"test_per_fault/fault_{i:02d}_X.npy") for i in range(1, 13)
        ]
        fault_labels = [label for label, _ in fault_options]
        selected_label = st.selectbox("테스트 데이터 선택 (정상 또는 Fault 0~12)", fault_labels, help="분석할 Fault 데이터를 선택하세요")
        selected_path = dict(fault_options)[selected_label]
        data = None
        try:
            data = np.load(selected_path)
            st.success(f"{selected_label} 데이터 로드 완료: {data.shape}")
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
        if data is not None:
            st.markdown("---")
            st.subheader("🚀 파이프라인 실행")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_pipeline = st.button("🚀 파이프라인 실행", type="primary", use_container_width=True)
            if run_pipeline:
                st.subheader("🔍 파이프라인 실행 중...")
                with st.spinner("파이프라인을 실행하고 있습니다..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    steps = ["Model1: Fault 탐지", "Model2: 조작 변수 정상화", "Model3: 반응 변수 예측", "Model4: 정상 여부 재분류"]
                    for i, step in enumerate(steps):
                        status_text.text(f"진행 중: {step}")
                        progress_bar.progress((i + 1) * 25)
                        time.sleep(0.5)
                    results = run_tep_pipeline(data)
                    progress_bar.progress(100)
                    status_text.text("완료!")
                    display_results(results)

if __name__ == "__main__":
    main() 