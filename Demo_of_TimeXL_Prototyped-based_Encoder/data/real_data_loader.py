
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

class WeatherTimeXLDataset(Dataset):
    """
    Kaggle Historical Hourly Weather Dataset 加载器
    
    任务：基于过去 N 小时的温度序列，预测当前的天气描述 (Classification Task)
    
    输入:
    - 时间序列: 过去 `seq_length` 小时的温度 (Temperature)
    - 文本: 对应的天气描述 (Weather Description) 作为辅助或目标
    
    标签:
    - 简单的天气分类 (e.g., 'sky is clear', 'rain', 'clouds')
    """
    def __init__(self, data_dir: str, city: str = 'San Francisco', seq_length: int = 24, split: str = 'train', simplify_labels: bool = True):
        self.data_dir = data_dir
        self.city = city
        self.seq_length = seq_length
        self.split = split
        self.simplify_labels = simplify_labels
        
        # Load raw data
        self.temp_df = pd.read_csv(os.path.join(data_dir, 'temperature.csv'))
        self.desc_df = pd.read_csv(os.path.join(data_dir, 'weather_description.csv'))
        
        # Preprocess
        self.data, self.texts, self.labels, self.label_map = self._preprocess()

    def _get_simplified_label(self, desc: str) -> str:
        """Map detailed description to simplified category"""
        desc = str(desc).lower()
        
        # Snow / Ice categories
        if any(x in desc for x in ['snow', 'sleet', 'ice', 'hail', 'freezing']):
            return 'Snow'
            
        # Rain / Precipitation categories
        if any(x in desc for x in ['rain', 'drizzle', 'shower', 'thunderstorm', 'storm', 'squall']):
            return 'Rain'
            
        # Everything else (Clear, Clouds, Fog, Mist, Haze, Smoke, etc.)
        return 'No Precipitation'

    def _preprocess(self) -> Tuple[torch.Tensor, List[str], torch.Tensor, dict]:
        # 1. Extract city data
        # Fill missing values with forward fill
        temps = self.temp_df[self.city].ffill().bfill().values
        descs = self.desc_df[self.city].ffill().bfill().values
        
        # 2. Filter valid data (remove remaining NaNs if any)
        # Convert temps to float (Kelvin to Celsius optional, keep normalized)
        temps = (temps - np.mean(temps)) / (np.std(temps) + 1e-8)
        
        # 3. Create samples
        # We need sequences of length `seq_length`
        # Target is the description at the END of the sequence
        
        data_list = []
        text_list = []
        label_list = []
        
        # Create label mapping
        if self.simplify_labels:
            # Map all descriptions to simplified ones first
            simplified_descs = [self._get_simplified_label(d) for d in descs]
            unique_labels = sorted(list(set(simplified_descs)))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            # Use simplified descriptions for processing
            process_descs = simplified_descs
        else:
            unique_labels = sorted(list(set(descs)))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            process_descs = descs
        
        # Create windows
        num_samples = len(temps) - self.seq_length
        
        # Split indices
        split_idx = int(num_samples * 0.8)
        val_idx = int(num_samples * 0.9)
        
        if self.split == 'train':
            indices = range(0, split_idx)
        elif self.split == 'val':
            indices = range(split_idx, val_idx)
        else: # test
            indices = range(val_idx, num_samples)
            
        for i in indices:
            # Time series window
            window = temps[i : i + self.seq_length]
            
            # Target description (at the last step)
            original_desc = descs[i + self.seq_length - 1]
            target_label_str = process_descs[i + self.seq_length - 1]
            
            # For this task, we'll use the target description as the "text" input as well 
            # We keep the original detailed description in text to give LLM more context, 
            # even if classification target is simplified.
            text_input = f"The weather condition is {original_desc}."
            if self.simplify_labels:
                text_input += f" Simplified category: {target_label_str}."
            
            data_list.append(window)
            text_list.append(text_input)
            label_list.append(label_map[target_label_str])
            
        # Convert to tensors
        # Data shape: [N, 1, T] (1 feature: temperature)
        data_tensor = torch.tensor(np.array(data_list), dtype=torch.float32).unsqueeze(1)
        labels_tensor = torch.tensor(label_list, dtype=torch.long)
        
        return data_tensor, text_list, labels_tensor, label_map
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.texts[idx], self.labels[idx]

class ProcessedWeatherDataset(Dataset):
    """
    加载预处理后的概率分布标签数据集
    """
    def __init__(self, data_dir: str, city: str = 'San Francisco', split: str = 'train'):
        self.data_dir = data_dir
        self.city = city.replace(' ', '_')
        self.split = split
        
        # Load processed data
        filename = f'processed_{split}_{self.city}.pt'
        filepath = os.path.join(data_dir, 'processed_data', filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed file not found: {filepath}")
            
        print(f"Loading {filepath}...")
        self.samples = torch.load(filepath)
        
        # Hardcoded label map based on preprocessing logic
        self.label_map = {'No Precipitation': 0, 'Rain': 1, 'Snow': 2}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # numerical_x: [24, 5]
        # We need to adapt it to typical model input [C, T] or keep as [T, C]
        # AdaptedMockModel expects flattened input, so shape matters less as long as it's consistent.
        # But for CNNs/RNNs, usually [Seq, Features] or [Features, Seq].
        # Let's return as is: [24, 5]
        
        # text_x: List[str] length 24. 
        # We need to aggregate them into a single string or return list.
        # Trainer expects a list of strings for the batch, so for a single sample we should return a string?
        # Or if the encoder handles list of list of strings?
        # BaseTrainer: `refined_texts = original_texts` -> `z_text = self.model.text_encoder(refined_texts)`
        # TrainableTextEncoder._tokenize expects `texts` (list of strings).
        # If we return a list of 24 strings here, the batch collation will make it a list of lists?
        # Actually, let's concat the descriptions into one string for simplicity, 
        # or just take the last one?
        # The prompt says "text_x: List[str] length 24".
        # Let's concat them with spaces.
        text_list = sample['text_x']
        # Maybe "hour 0: rain. hour 1: clear..."? Or just join.
        text_input = " ".join(text_list)
        
        # label_y: [3] (Distribution)
        label = sample['label_y']
        
        return sample['numerical_x'], text_input, label

def get_weather_dataloader(data_dir, city='San Francisco', batch_size=32, split='train', simplify_labels=True):
    dataset = WeatherTimeXLDataset(data_dir, city=city, split=split, simplify_labels=simplify_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))

def get_processed_dataloader(data_dir, city='San Francisco', batch_size=32, split='train'):
    dataset = ProcessedWeatherDataset(data_dir, city=city, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
