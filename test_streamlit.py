#!/usr/bin/env python3
"""
Streamlit 앱 테스트 스크립트
"""

import subprocess
import sys
import os

def test_imports():
    """필수 모듈 임포트 테스트"""
    print("🔍 모듈 임포트 테스트 중...")
    
    try:
        import streamlit as st
        print("✅ Streamlit 임포트 성공")
    except ImportError as e:
        print(f"❌ Streamlit 임포트 실패: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy 임포트 성공")
    except ImportError as e:
        print(f"❌ NumPy 임포트 실패: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly 임포트 성공")
    except ImportError as e:
        print(f"❌ Plotly 임포트 실패: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch 임포트 성공")
    except ImportError as e:
        print(f"❌ PyTorch 임포트 실패: {e}")
        return False
    
    return True

def test_file_structure():
    """파일 구조 테스트"""
    print("\n📁 파일 구조 테스트 중...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        ".streamlit/config.toml",
        "README_streamlit.md"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 존재")
        else:
            print(f"❌ {file_path} 없음")
            return False
    
    return True

def test_data_files():
    """데이터 파일 테스트"""
    print("\n📊 데이터 파일 테스트 중...")
    
    data_files = [
        "data/final_X.npy",
        "data/final_Y.npy",
        "model_pretrained/model1/30_epoch_checkpoint.pth"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 존재")
        else:
            print(f"⚠️ {file_path} 없음 (선택사항)")
    
    return True

def run_streamlit():
    """Streamlit 앱 실행"""
    print("\n🚀 Streamlit 앱 실행 중...")
    print("브라우저에서 http://localhost:8501 접속")
    print("종료하려면 Ctrl+C")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit 앱 종료")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit 실행 실패: {e}")

def main():
    """메인 함수"""
    print("🏭 Tennessee Eastman Process - Streamlit 테스트")
    print("=" * 50)
    
    # 1. 모듈 임포트 테스트
    if not test_imports():
        print("\n❌ 모듈 임포트 실패. requirements.txt를 확인하세요.")
        return
    
    # 2. 파일 구조 테스트
    if not test_file_structure():
        print("\n❌ 필수 파일이 없습니다.")
        return
    
    # 3. 데이터 파일 테스트
    test_data_files()
    
    # 4. Streamlit 실행
    print("\n✅ 모든 테스트 통과!")
    run_streamlit()

if __name__ == "__main__":
    main() 