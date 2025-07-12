import google.generativeai as genai
import os
from typing import Dict, List, Tuple, Any

# 현재 파일의 절대 경로를 가져와서 API 키 파일 경로 설정
current_file_abs_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_abs_path)
key_path = os.path.join(current_dir, 'MY_KEY.txt')

# API 키 파일에서 읽기
with open(key_path, 'r', encoding='utf-8-sig') as f:
    api_key = f.read().strip()
API_KEY = api_key

class LLM:
    def __init__(self, api_key: str = None):
        """
        LLM 클라이언트 초기화
        
        Args:
            api_key: Gemini API 키. None이면 MY_KEY.txt에서 로드
        """
        if api_key is None:
            api_key = API_KEY
            if not api_key or api_key == "your_gemini_api_key_here":
                raise ValueError("MY_KEY.txt 파일에 유효한 Gemini API 키를 설정해주세요.")
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    
    def load_prompt_from_file(self, prompt_file: str) -> str:
        """
        txt 파일에서 프롬프트를 로드합니다.
        
        Args:
            prompt_file: 프롬프트 파일 경로
            
        Returns:
            프롬프트 내용
        """
        prompt_path = os.path.join(current_dir, 'prompts', prompt_file)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return ""
    
    def generate_response(self, prompt: str, system_prompt_file: str = None) -> str:
        """
        LLM에 프롬프트를 전송하고 응답을 받습니다.
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt_file: 시스템 프롬프트 파일명 (선택사항)
            
        Returns:
            모델의 응답 텍스트
        """
        try:
            if system_prompt_file:
                system_message = self.load_prompt_from_file(system_prompt_file)
                response = self.model.generate_content([
                    {"role": "user", "parts": [system_message + "\n\n" + prompt]}
                ])
            else:
                response = self.model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"LLM API 호출 중 오류 발생: {e}")
            return f"오류: {str(e)}"
    
    def send_prompt_to_model(self, prompt_file: str) -> str:
        """
        txt 파일의 프롬프트를 모델에게 전송합니다.
        
        Args:
            prompt_file: 프롬프트 파일명 (예: 'base_prompt.txt')
            
        Returns:
            모델의 응답
        """
        # 프롬프트 로드
        prompt = self.load_prompt_from_file(prompt_file)
        return self.generate_response(prompt)
    
    def send_multiple_prompts_to_model(self, prompt_files: List[str]) -> str:
        """
        여러 txt 파일의 프롬프트를 모델에게 전송합니다.
        
        Args:
            prompt_files: 프롬프트 파일명 리스트 (예: ['base_prompt.txt', 'dataset_prompt.txt'])
            
        Returns:
            모델의 응답
        """
        # 모든 프롬프트 로드 및 결합
        combined_prompt = ""
        for prompt_file in prompt_files:
            prompt = self.load_prompt_from_file(prompt_file)
            combined_prompt += prompt + "\n\n"
        
        return self.generate_response(combined_prompt)