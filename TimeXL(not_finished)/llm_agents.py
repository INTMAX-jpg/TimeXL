import time
import logging
import os
import requests
from typing import Optional, Dict, Any, List

class BaseLLMAgent:
    """LLM智能体基类"""
    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model_name = model_name
        
        # 优先使用传入的 api_key，否则尝试从环境变量读取
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key and model_name != "mock":
            # 仅记录警告，不直接报错，以便 MockAgent 可以正常工作
            logging.warning("Environment variable DEEPSEEK_API_KEY not set. API calls will fail if no key is provided.")
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """生成响应 (DeepSeek API)"""
        if not self.api_key:
            raise ValueError("Please set environment variable DEEPSEEK_API_KEY or provide it in __init__")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        # Merge extra kwargs into data (e.g., max_tokens)
        data.update(kwargs)

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status() # Raise error for 4xx/5xx status codes
            
            result = response.json()
            # Extract content from response
            # DeepSeek response format is compatible with OpenAI
            content = result['choices'][0]['message']['content']
            return content
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response content: {e.response.text}")
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"Failed to parse API response: {e}")
            self.logger.error(f"Response json: {result}")
            raise

class PredictionLLM(BaseLLMAgent):
    """预测LLM智能体"""
    def predict(self, text: str, case_explanations: List[str], task_instruction: str = "Predict the time series trend") -> str:
        """基于文本和案例解释进行预测"""
        prompt = self._build_prediction_prompt(text, case_explanations, task_instruction)
        return self.generate(prompt)

    def _build_prediction_prompt(self, text: str, case_explanations: List[str], task_instruction: str) -> str:
        explanations_str = "\n".join(case_explanations)
        return f"""Task: You are a meteorological expert. Analyze the sequence of weather descriptions recorded over the past 24 hours. Your goal is to estimate the proportion of hours in the NEXT 24 hours that will fall into each of the following three categories:
No Precipitation
Rain
Snow

Input Format: A chronological list of weather descriptions observed every hour for the last day.

Output Requirements: 
Provide your estimate as three probabilities (0.0 to 1.0) corresponding to the fraction of time each condition is expected to occur. The three probabilities must sum to 1.0.

Input Text: 
{text}

Similar Cases (Reference):
{explanations_str}

Prediction:"""

class ReflectionLLM(BaseLLMAgent):
    """反思LLM智能体"""
    def reflect(self, ground_truth: Any, prediction: Any, text: str) -> str:
        """生成反思反馈"""
        prompt = self._build_reflection_prompt(ground_truth, prediction, text)
        return self.generate(prompt)

    def _build_reflection_prompt(self, ground_truth: Any, prediction: Any, text: str) -> str:
        return f"Task: Analyze the prediction error.\nText: {text}\nPrediction: {prediction}\nGround Truth: {ground_truth}\n\nProvide a reflection on what information was missing or misleading in the text to guide future improvements:"

class RefinementLLM(BaseLLMAgent):
    """优化LLM智能体"""
    def refine(self, reflection: str, text: str) -> str:
        """优化文本"""
        prompt = self._build_refinement_prompt(reflection, text)
        return self.generate(prompt)

    def _build_refinement_prompt(self, reflection: str, text: str) -> str:
        return f"Task: Refine the text based on the reflection to improve prediction accuracy.\n\nOriginal Text: {text}\n\nReflection: {reflection}\n\nRefined Text:"

class MockLLMAgent(BaseLLMAgent):
    """模拟LLM用于测试"""
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        super().__init__(model_name="mock")
        self.responses = responses or {}
        self.call_history = []
    
    def predict(self, text: str, case_explanations: List[str], task_instruction: str = "Predict the time series trend") -> str:
        # 模拟预测逻辑
        self.call_history.append(('predict', text))
        return "Rain"  # 模拟预测结果 (Update to reasonable weather output)
    
    def reflect(self, ground_truth: Any, prediction: Any, text: str) -> str:
        # 模拟反思反馈
        self.call_history.append(('reflect', text))
        return "The text missed key volatility indicators."
    
    def refine(self, reflection: str, text: str) -> str:
        # 模拟文本优化
        self.call_history.append(('refine', text))
        return f"{text} [Refined based on: {reflection}]"
    
    def generate(self, prompt: str, **kwargs) -> str:
        return "Mock generation"
