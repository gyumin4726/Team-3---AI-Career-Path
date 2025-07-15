import os

try:
    import streamlit as st
    llm_key = st.secrets["llm"]["api_key"]
except (ImportError, KeyError):
    llm_key = os.environ.get("LLM_API_KEY", None)
    if not llm_key:
        # txt 파일에서 읽기 (fallback)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        key_path = os.path.join(current_dir, 'MY_KEY.txt')
        if os.path.exists(key_path):
            with open(key_path, 'r', encoding='utf-8-sig') as f:
                llm_key = f.read().strip()

import google.generativeai as genai
from typing import Dict, List, Tuple, Any
import re

# 변수명-설명 매핑 유틸 함수
_x_var_map = None
_m_var_map = None

def get_variable_maps():
    global _x_var_map, _m_var_map
    if _x_var_map is not None and _m_var_map is not None:
        return _x_var_map, _m_var_map
    # dataset_prompt.txt 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, 'prompts', 'dataset_prompt.txt')
    x_map = {}
    m_map = {}
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 조작 변수(m)
        m_section = False
        x_section = False
        for line in lines:
            line = line.strip()
            if line.startswith('조작 변수 (m)'):
                m_section = True
                x_section = False
                continue
            if line.startswith('반응 변수 (x)'):
                x_section = True
                m_section = False
                continue
            if m_section and line.startswith('- m'):
                m = re.match(r'- (m\d+): ([^\(]+)\(([^\)]+)\)', line)
                if m:
                    idx = int(m.group(1)[1:])
                    name = m.group(2).strip()
                    desc = m.group(3).strip()
                    m_map[idx] = f"m{idx}: {name}({desc})"
            if x_section and line.startswith('- x'):
                m = re.match(r'- (x\d+): ([^\(]+)\(([^\)]+)\)', line)
                if m:
                    idx = int(m.group(1)[1:])
                    name = m.group(2).strip()
                    desc = m.group(3).strip()
                    x_map[idx] = f"x{idx}: {name}({desc})"
                else:
                    # x10, x21, x22 등 단위만 있는 경우
                    m2 = re.match(r'- (x\d+): ([^\[]+)\[([^\]]+)\]', line)
                    if m2:
                        idx = int(m2.group(1)[1:])
                        name = m2.group(2).strip()
                        desc = m2.group(3).strip()
                        x_map[idx] = f"x{idx}: {name} [{desc}]"
    _x_var_map = x_map
    _m_var_map = m_map
    return x_map, m_map

class LLM:
    def __init__(self):
        if not llm_key:
            raise ValueError("LLM API 키가 필요합니다. (st.secrets['llm']['api_key'], 환경변수 LLM_API_KEY, 또는 MY_KEY.txt 중 하나)")
        self.api_key = llm_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    def load_prompt_from_file(self, prompt_file: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, 'prompts', prompt_file)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return ""

    def generate_response(self, prompt: str, system_prompt_file: str = None) -> str:
        try:
            # 항상 base_prompt.txt, dataset_prompt.txt, fault_list.txt를 system prompt로 포함
            base_prompt = self.load_prompt_from_file('base_prompt.txt')
            dataset_prompt = self.load_prompt_from_file('dataset_prompt.txt')
            fault_list_prompt = self.load_prompt_from_file('fault_list.txt')
            system_message = base_prompt + "\n\n" + dataset_prompt + "\n\n" + fault_list_prompt
            # 추가 system_prompt_file이 있으면 덧붙임
            if system_prompt_file:
                extra_system = self.load_prompt_from_file(system_prompt_file)
                system_message += "\n\n" + extra_system
            full_prompt = system_message + "\n\n" + prompt
            #try:
            #    import streamlit as st
            #    st.write('---')
            #    st.write('**[LLM 디버그] 실제 전달 프롬프트:**')
            #    st.write(full_prompt)
            #    st.write('---')
            #except Exception:
            #    pass
            response = self.model.generate_content([
                {"role": "user", "parts": [full_prompt]}
            ])
            return response.text
        except Exception as e:
            print(f"LLM API 호출 중 오류 발생: {e}")
            return f"오류: {str(e)}"

    def send_prompt_to_model(self, prompt_file: str) -> str:
        prompt = self.load_prompt_from_file(prompt_file)
        return self.generate_response(prompt)

    def send_multiple_prompts_to_model(self, prompt_files: List[str]) -> str:
        combined_prompt = ""
        for prompt_file in prompt_files:
            prompt = self.load_prompt_from_file(prompt_file)
            combined_prompt += prompt + "\n\n"
        return self.generate_response(combined_prompt)