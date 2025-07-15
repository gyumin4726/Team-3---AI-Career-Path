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

# CSS 스타일 (더 세련되고 현대적으로 개선)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Noto Sans KR', sans-serif !important;
}
.main-header {
    font-size: 2.8rem;
    color: #fff;
    background: linear-gradient(90deg, #1f77b4 0%, #43cea2 100%);
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.2rem 0 1.2rem 0;
    border-radius: 1.2rem;
    box-shadow: 0 4px 16px rgba(30,80,180,0.08);
}
.metric-card {
    background: linear-gradient(90deg, #f8fafc 60%, #e0f7fa 100%);
    padding: 1.2rem 1rem;
    border-radius: 1rem;
    box-shadow: 0 2px 8px rgba(30,80,180,0.07);
    margin: 0.7rem 0;
    text-align: center;
}
.metric-title {
    font-size: 1.1rem;
    color: #1f77b4;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 1.7rem;
    color: #222;
    font-weight: 700;
}
.result-section {
    margin-top: 2.2rem;
    margin-bottom: 2.2rem;
    padding: 1.5rem 1rem;
    background: #f0f4f8;
    border-radius: 1.2rem;
    box-shadow: 0 2px 8px rgba(30,80,180,0.07);
}
.stButton > button {
    background: linear-gradient(90deg, #1f77b4 0%, #43cea2 100%);
    color: white;
    font-weight: 700;
    border-radius: 0.7rem;
    border: none;
    padding: 0.7rem 0;
    font-size: 1.1rem;
    transition: 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #43cea2 0%, #1f77b4 100%);
    color: #fff;
    box-shadow: 0 2px 8px rgba(30,80,180,0.13);
}
/* 컴팩트 metric 카드 */
.compact-metric {
    padding: 0.45rem 0.25rem !important;
    margin: 0.15rem 0 !important;
    min-width: 80px;
    max-width: 120px;
}
.compact-metric .metric-title {
    font-size: 0.85rem !important;
    margin-bottom: 0.05rem !important;
}
.compact-metric .metric-value {
    font-size: 1.01rem !important;
}
/* Expander 타이틀(분석 설명) 더 작고 가운데 정렬 */
.streamlit-expanderHeader {
    font-size: 1.01rem !important;
    text-align: center !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
}
/* 컴팩트 selectbox */
.compact-select .stSelectbox {
    max-width: 120px !important;
    min-width: 80px !important;
    width: 120px !important;
    margin: 0 auto !important;
}
.compact-select .stSelectbox > div[data-baseweb="select"] {
    min-height: 28px !important;
    font-size: 0.93rem !important;
    padding: 0 4px !important;
}
.compact-select .stSelectbox input {
    min-width: 60px !important;
    max-width: 100px !important;
    font-size: 0.93rem !important;
    padding: 2px 4px !important;
}
.compact-select .stSelectbox [data-baseweb="select"] > div {
    min-height: 28px !important;
    padding: 0 4px !important;
}
/* 결함 시점 카드만 더 크게 */
.wide-metric {
    max-width: 190px !important;
    min-width: 150px !important;
}
</style>
""", unsafe_allow_html=True)

def original_time_to_window_index(original_time, window_size, step_size):
    window_num = original_time // step_size
    timestep_in_window = original_time % step_size
    window_index = window_num * window_size + timestep_in_window
    return window_index

def run_tep_pipeline(data: np.ndarray):
    """TEP 파이프라인 실행"""
    try:
        # 파이프라인 초기화
        pipeline = TEPPipeline()
        # 파이프라인 실행
        results = pipeline.run_full_pipeline(data)
        return results, pipeline  # pipeline 객체도 반환
    except Exception as e:
        st.error(f"파이프라인 실행 오류: {e}")
        return None, None

def display_results(results: Dict[str, Any]):
    """결과 표시"""
    if results is None:
        return
    st.markdown('<h3 style="text-align:center; margin-top:1.2rem;">📊 분석 결과</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1,2,3,2,1])
    with col2:
        st.markdown('<div class="metric-card compact-metric">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">초기 결함 유형</div><div class="metric-value">{results.get("model1_fault_class", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        fault_time = results.get('model1_fault_time', None)
        st.markdown('<div class="metric-card compact-metric wide-metric">', unsafe_allow_html=True)
        if fault_time is not None:
            total_minutes = fault_time * 3
            hours = total_minutes // 60
            minutes = total_minutes % 60
            st.markdown(f'<div class="metric-title">결함 시점</div><div class="metric-value">{total_minutes}분, {hours}시간 {minutes}분</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-title">결함 시점</div><div class="metric-value">없음</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        success = results.get('success', False)
        status = "✅ 성공" if success else "❌ 실패"
        st.markdown('<div class="metric-card compact-metric">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">정상화 상태</div><div class="metric-value">{status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # 파이프라인 상태
    pipeline_status = results.get('pipeline_status', 'unknown')
    if pipeline_status == 'early_termination_normal':
        st.markdown('<div style="color:#219150;font-size:1.15rem;font-weight:700; text-align:center;">🟢 정상 상태로 감지되어 파이프라인이 조기 종료되었습니다.</div>', unsafe_allow_html=True)
    elif 'normalized' in pipeline_status:
        iterations = results.get('iterations', 0)
        st.markdown(f'<div style="color:#f7b731;font-size:1.15rem;font-weight:700; text-align:center;">🌟 {iterations}회 반복 후 정상화에 성공했습니다!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#d7263d;font-size:1.15rem;font-weight:700; text-align:center;">🛑 최대 반복 횟수 초과로 정상화에 실패했습니다.</div>', unsafe_allow_html=True)
    # LLM 설명
    if 'llm_explanations' in results:
        st.markdown('<h3 style="text-align:center; margin-top:1.2rem;">🤖 AI 분석 설명</h3>', unsafe_allow_html=True)
        explanations = results['llm_explanations']
        if 'model1' in explanations:
            with st.expander("Model1 (Fault 탐지 + 분류) 분석", expanded=False):
                st.write(explanations['model1'])
        if 'model2' in explanations:
            with st.expander("Model2 (조작 변수 정상화) 분석", expanded=False):
                st.write(explanations['model2'])
        if 'model3' in explanations:
            with st.expander("Model3 (반응 변수 예측) 분석", expanded=False):
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

def plot_single_m_change(original_data, normalized_m, m_index, fault_time, height=600):
    """
    Model2 정상화 전후 변화가 큰 Top3 조작 변수의 시계열을 비교 시각화
    Args:
        original_data: (B, 50, 52) 원본 데이터
        normalized_m: (B, 50, 11) 정상화된 조작 변수
        m_index: 실제 변수 인덱스(41~51)
        fault_time: 결함 발생 시점(슬라이딩 윈도우 인덱스)
    Returns:
        plotly.graph_objects.Figure
    """
    # 조작 변수명 매핑
    m_names = {
        41: "m1: D 피드 유량 밸브",
        42: "m2: E 피드 유량 밸브", 
        43: "m3: A 피드 유량 밸브",
        44: "m4: 총 피드 스트리퍼 유량 밸브",
        45: "m5: 압축기 순환 밸브",
        46: "m6: 퍼지 밸브",
        47: "m7: 분리기 액체 유출 밸브",
        48: "m8: 스트리퍼 액체 제품 유출 밸브",
        49: "m9: 스트리퍼 증기 밸브",
        50: "m10: 반응기 냉각수 유량 밸브",
        51: "m11: 응축기 냉각수 유량 밸브"
    }
    
    B, T, S = original_data.shape
    orig_m = original_data[:, :, 41:]  # (B, 50, 11)
    orig_m_2d = orig_m.reshape(B * T, 11)
    norm_m_2d = normalized_m.reshape(B * T, 11)
    m_idx = m_index - 41  # 0~10
    m_name = m_names.get(m_index, f"조작 변수 {m_index}")
    colors = px.colors.qualitative.Set1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(orig_m_2d.shape[0])),
        y=orig_m_2d[:, m_idx],
        mode='lines',
        name=f'원본 {m_name}',
        line=dict(color=colors[0]),
        legendgroup='orig',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=list(range(norm_m_2d.shape[0])),
        y=norm_m_2d[:, m_idx],
        mode='lines',
        name=f'정상화 {m_name}',
        line=dict(color=colors[1]),
        legendgroup='norm',
        showlegend=True
    ))

    fig.update_layout(
        height=height,
        xaxis_title="슬라이딩 윈도우 시점 (0~4599)",
        yaxis_title="조작 변수 값 (%)",
        showlegend=True,
        title=m_name
    )
    return fig

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
            <div style="background: linear-gradient(135deg, #e0f7fa 60%, #b2ebf2 100%); border-radius: 22px; box-shadow: 0 4px 16px rgba(30,80,180,0.10); padding: 38px 22px; min-width: 220px; text-align: center; border: 2px solid #1f77b4;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">👥</div>
                <h2 style="font-size: 1.35rem; font-weight: 800; color: #1f77b4; margin-bottom: 1rem; letter-spacing:-1px;">팀 정보</h2>
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
            <div style="background: linear-gradient(135deg, #fffde4 60%, #f7e8c3 100%); border-radius: 22px; box-shadow: 0 4px 16px rgba(255,193,7,0.10); padding: 38px 22px; min-width: 240px; max-width: 340px; margin: 0 auto; text-align: center; border: 2px solid #ffc107;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">📊</div>
                <h2 style="font-size: 1.35rem; font-weight: 800; color: #ffc107; margin-bottom: 1rem; letter-spacing:-1px;">데이터셋 정보</h2>
                <ul style="font-size: 1.05rem; line-height: 2; color: #333; font-weight: 500; text-align: left; margin-left: 1.2em;">
                    <li><b>센서 개수:</b> 52개 (22개 공정, 19개 분석, 11개 조작)</li>
                    <li><b>결함 유형:</b> 12가지 (정상 포함)</li>
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
            <div style="background: linear-gradient(135deg, #e1f8e6 60%, #b7e7c2 100%); border-radius: 22px; box-shadow: 0 4px 16px rgba(40,167,69,0.10); padding: 38px 22px; min-width: 240px; max-width: 360px; margin: 0 auto; text-align: center; border: 2px solid #28a745;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">🔗</div>
                <h2 style="font-size: 1.35rem; font-weight: 800; color: #28a745; margin-bottom: 1rem; letter-spacing:-1px;">파이프라인 원리</h2>
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
        st.markdown('<h2 style="text-align:center; margin-top:1.5rem; margin-bottom:1.2rem;">파이프라인 분석</h2>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align:center; margin-top:1.2rem;">📊 데이터 로드</h3>', unsafe_allow_html=True)
        # test_per_fault 폴더의 정상, 0~12 fault 선택지 제공
        fault_options = [
            ("정상", "test_per_fault/fault_00_X.npy")
        ] + [
            (f"Fault {i:02d}", f"test_per_fault/fault_{i:02d}_X.npy") for i in range(1, 13)
        ]
        fault_labels = [label for label, _ in fault_options]
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown('<div style="text-align:center; font-size:1.08rem; font-weight:600; margin-bottom:0.5rem;">테스트 데이터 선택 (정상 또는 Fault 0~12)</div>', unsafe_allow_html=True)
            st.markdown('<div class="compact-select">', unsafe_allow_html=True)
            selected_label = st.selectbox("", fault_labels, help="분석할 Fault 데이터를 선택하세요")
            st.markdown('</div>', unsafe_allow_html=True)
        selected_path = dict(fault_options)[selected_label]
        data = None
        try:
            data = np.load(selected_path)
            st.markdown(
                f'<div style="text-align:center; font-size:1.13rem; font-weight:600; color:#219150; margin:1rem 0 1.2rem 0;">'
                f'✅ {selected_label} 시뮬레이션 로드 완료'
                f'</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
        if data is not None:
            st.markdown("---")
            st.markdown('<h3 style="text-align:center; margin-top:1.2rem;">🚀 파이프라인 실행</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_pipeline = st.button("🚀 파이프라인 실행", type="primary", use_container_width=True)
            if run_pipeline:
                col_a, col_b, col_c = st.columns([1,2,1])
                with col_b:
                    execution_status = st.empty()
                    execution_status.markdown('<div style="text-align:center; font-size:1.02rem; margin-top:0.7rem; margin-bottom:0.3rem;">🔍 파이프라인 실행 중...</div>', unsafe_allow_html=True)
                    with st.spinner("파이프라인을 실행하고 있습니다..."):
                        progress_bar = st.empty()
                        progress_bar.progress(0)
                        status_text = st.empty()
                        steps = ["Model1 - Fault 탐지", "Model2 - 조작 변수 정상화", "Model3 - 반응 변수 예측", "Model4 - 정상 여부 재분류"]
                        for i, step in enumerate(steps):
                            status_text.markdown(f"<div style='text-align:center; font-size:0.97rem; margin-bottom:0.1rem;'>{step}</div>", unsafe_allow_html=True)
                            progress_bar.progress((i + 1) * 25)
                            time.sleep(1.1)
                        results, pipeline = run_tep_pipeline(data)  # pipeline 객체도 받음
                        progress_bar.progress(100)
                        status_text.markdown("")  # 진행 단계 텍스트 지우기
                        progress_bar.empty()  # 프로그레스 바 완전히 제거
                        execution_status.markdown('<div style="text-align:center; font-size:1.02rem; margin-top:0.7rem; margin-bottom:0.3rem;"></div>', unsafe_allow_html=True)
                        display_results(results)
                        # 결과를 세션 상태에 저장
                        st.session_state['tep_results'] = results
                        st.session_state['tep_input_data'] = data
                        # Model2 Top3 인덱스/통계도 저장 (정상화가 반영된 pipeline 사용)
                        if results is not None and results.get('normalized_m') is not None and pipeline is not None:
                            model2_llm_result = pipeline.model2_module.get_results_for_llm(
                                pipeline.original_input_data, pipeline.first_fault_time)
                            st.session_state['model2_top3_indices'] = model2_llm_result.get('top3_indices', [41, 42, 43])
                            st.session_state['model2_top3_stats'] = model2_llm_result.get('stats', {})
                    
    with tab3:
        # --- 파이프라인 실행 결과 기반 Model2 Top3 조작 변수 변화 시각화 ---
        if 'tep_results' in st.session_state and st.session_state['tep_results'] is not None:
            results = st.session_state['tep_results']
            input_data = st.session_state.get('tep_input_data', None)
            # 아래에서 Top3 인덱스/통계는 세션에 저장된 값을 사용
            top3_indices = st.session_state.get('model2_top3_indices', [41, 42, 43])
            top3_stats = st.session_state.get('model2_top3_stats', {})
            if results.get('normalized_m') is not None and input_data is not None:
                fault_time = results.get('model1_fault_time', None)
                st.markdown('<h3 style="text-align:center; margin-top:1.2rem;">Model2 정상화 전후 Top3 조작 변수 변화</h3>', unsafe_allow_html=True)

                # 세션 상태에 현재 인덱스 저장
                if 'current_plot_idx' not in st.session_state:
                    st.session_state['current_plot_idx'] = 0
                idx = st.session_state['current_plot_idx']
                # 인덱스 범위 제한
                idx = max(0, min(idx, 2))
                st.session_state['current_plot_idx'] = idx

                # 단일 subplot만 그리기
                def plot_single_m_change(original_data, normalized_m, m_index, fault_time, height=600):
                    m_names = {
                        41: "m1: D 피드 유량 밸브",
                        42: "m2: E 피드 유량 밸브",
                        43: "m3: A 피드 유량 밸브",
                        44: "m4: 총 피드 스트리퍼 유량 밸브",
                        45: "m5: 압축기 순환 밸브",
                        46: "m6: 퍼지 밸브",
                        47: "m7: 분리기 액체 유출 밸브",
                        48: "m8: 스트리퍼 액체 제품 유출 밸브",
                        49: "m9: 스트리퍼 증기 밸브",
                        50: "m10: 반응기 냉각수 유량 밸브",
                        51: "m11: 응축기 냉각수 유량 밸브"
                    }
                    B, T, S = original_data.shape
                    orig_m = original_data[:, :, 41:]  # (B, 50, 11)
                    orig_m_2d = orig_m.reshape(B * T, 11)
                    norm_m_2d = normalized_m.reshape(B * T, 11)
                    m_idx = m_index - 41
                    m_name = m_names.get(m_index, f"조작 변수 {m_index}")
                    colors = px.colors.qualitative.Set1
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(orig_m_2d.shape[0])),
                        y=orig_m_2d[:, m_idx],
                        mode='lines',
                        name=f'원본 {m_name}',
                        line=dict(color=colors[0]),
                        legendgroup='orig',
                        showlegend=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(norm_m_2d.shape[0])),
                        y=norm_m_2d[:, m_idx],
                        mode='lines',
                        name=f'정상화 {m_name}',
                        line=dict(color=colors[1]),
                        legendgroup='norm',
                        showlegend=True
                    ))
                    # fault_time은 이미 슬라이딩 윈도우 인덱스이므로 변환 없이 바로 사용
                    if fault_time is not None and isinstance(fault_time, (int, float)):
                        fault_time = original_time_to_window_index(fault_time, 50, 10)
                        fig.add_vline(x=int(fault_time), line_width=2, line_dash="dash", line_color="blue",
                                      annotation_text="이상 발생 시점", annotation_position="top right")
                    fig.update_layout(
                        height=height,
                        xaxis_title="슬라이딩 윈도우 시점 (0~4599)",
                        yaxis_title="조작 변수 값 (%)",
                        showlegend=True,
                        title=m_name
                    )
                    return fig

                # 캐러셀 스타일 시각화 UI
                st.markdown("""
                <style>
                .carousel-btn {
                    position: absolute;
                    top: 50%;
                    transform: translateY(-50%);
                    background: rgba(44,62,80,0.85);
                    color: #fff;
                    border: none;
                    border-radius: 50%;
                    width: 38px;
                    height: 38px;
                    font-size: 1.5rem;
                    font-weight: bold;
                    box-shadow: 0 2px 8px rgba(30,80,180,0.13);
                    cursor: pointer;
                    z-index: 10;
                    transition: background 0.2s;
                }
                .carousel-btn:disabled {
                    background: #ccc;
                    color: #eee;
                    cursor: not-allowed;
                }
                .carousel-dot {
                    display: inline-block;
                    width: 13px;
                    height: 13px;
                    margin: 0 5px;
                    background: #bbb;
                    border-radius: 50%;
                    transition: background 0.3s;
                }
                .carousel-dot.active {
                    background: #1f77b4;
                }
                .carousel-outer {
                    position: relative;
                    max-width: 900px;
                    margin: 0 auto 0.7rem auto;
                }
                </style>
                """, unsafe_allow_html=True)
                st.markdown('<div class="carousel-outer">', unsafe_allow_html=True)
                # 좌측 float 버튼
                col_btn_left, col_graph, col_btn_right = st.columns([1,8,1])
                with col_btn_left:
                    st.markdown('<div style="height: 260px;"></div>', unsafe_allow_html=True)  # 버튼 세로 정렬용
                    if st.button('❮', key='carousel_prev', help='이전', disabled=(idx==0)):
                        st.session_state['current_plot_idx'] = max(0, idx-1)
                with col_btn_right:
                    st.markdown('<div style="height: 260px;"></div>', unsafe_allow_html=True)
                    if st.button('❯', key='carousel_next', help='다음', disabled=(idx==2)):
                        st.session_state['current_plot_idx'] = min(2, idx+1)
                with col_graph:
                    st.plotly_chart(plot_single_m_change(input_data, results['normalized_m'], top3_indices[idx], fault_time, height=600), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                # 동그란 네비게이터
                nav_html = '<div style="text-align:center; margin-top: 0.5rem;">'
                for i in range(3):
                    active = 'active' if i == idx else ''
                    nav_html += f'<span class="carousel-dot {active}"></span>'
                nav_html += '</div>'
                st.markdown(nav_html, unsafe_allow_html=True)
                # Top3 통계도 아래에 출력 (원하면)
                st.write('Model2 Top3 indices:', top3_indices)
                st.write('Model2 mean_delta:', top3_stats)

if __name__ == "__main__":
    main() 