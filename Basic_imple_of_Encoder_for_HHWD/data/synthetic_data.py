
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

class SyntheticTimeXLDataset(Dataset):
    """
    合成数据集生成器，用于 TimeXL 模型的测试和开发。
    
    生成规则：
    - Class 0: 正弦波 (Sine Wave) + 文本 "Periodic pattern with smooth oscillations"
    - Class 1: 随机漫步 (Random Walk) + 文本 "Unpredictable random movement with trend"
    - Class 2: 阶跃信号 (Step Function) + 文本 "Sudden shifts in value levels"
    """
    def __init__(self, num_samples: int = 100, seq_length: int = 100, num_features: int = 5):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.classes = 3
        
        self.data, self.texts, self.labels = self._generate_data()
        
    def _generate_data(self) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        data = []
        texts = []
        labels = []
        
        base_texts = [
            "Periodic pattern with smooth oscillations indicating regular cycles.",
            "Unpredictable random movement with an upward trend and high volatility.",
            "Sudden shifts in value levels showing distinct regime changes."
        ]
        
        for i in range(self.num_samples):
            label = i % self.classes
            labels.append(label)
            texts.append(base_texts[label])
            
            # 生成时间序列数据 [N, T] -> 转置为 [T, N] 或保持 [N, T] 取决于编码器输入
            # 这里的 Encoder 期望 [B, N, T]
            sample_features = []
            
            for f in range(self.num_features):
                if label == 0: # Sine wave
                    t = np.linspace(0, 4*np.pi, self.seq_length)
                    noise = np.random.normal(0, 0.1, self.seq_length)
                    feature = np.sin(t + np.random.random()*np.pi) + noise
                    
                elif label == 1: # Random walk
                    steps = np.random.normal(0, 1, self.seq_length)
                    feature = np.cumsum(steps)
                    # Normalize
                    feature = (feature - feature.mean()) / (feature.std() + 1e-8)
                    
                else: # Step function
                    feature = np.zeros(self.seq_length)
                    idx1 = np.random.randint(self.seq_length // 4, self.seq_length // 2)
                    idx2 = np.random.randint(self.seq_length // 2, 3 * self.seq_length // 4)
                    feature[idx1:idx2] = 1.0
                    feature[idx2:] = -1.0
                    feature += np.random.normal(0, 0.1, self.seq_length)
                
                sample_features.append(feature)
            
            data.append(np.stack(sample_features))
            
        # Convert to tensors
        # Data shape: [num_samples, num_features, seq_length]
        data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return data_tensor, texts, labels_tensor
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 返回格式必须匹配 BaseTrainer 的解包逻辑: x_time, original_texts, targets
        return self.data[idx], self.texts[idx], self.labels[idx]

def get_synthetic_dataloader(num_samples=100, batch_size=16):
    dataset = SyntheticTimeXLDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
