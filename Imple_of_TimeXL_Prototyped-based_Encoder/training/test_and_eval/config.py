from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """
    训练配置类
    管理所有与训练、模型和LLM相关的超参数
    """
    # 基础训练参数
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10  # BaseTrainer 单次迭代的 epoch 数
    device: str = "cuda"
    seed: int = 42
    
    # 迭代优化参数 (Algorithm 1)
    max_iterations: int = 5
    alpha: float = 0.7
    save_dir: str = "./checkpoints"
    patience: int = 3  # 早停机制 patience
    
    # LLM 参数
    llm_model_name: str = "gpt-3.5-turbo"
    llm_api_key: Optional[str] = None
    
    # 数据参数
    max_seq_len: int = 100
    
    def __post_init__(self):
        """参数校验"""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
