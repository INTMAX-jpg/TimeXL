
import os
import sys
import torch
import logging
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)# 把根目录加入path，让下面的import能找到自定义模块

# Import components
from data.real_data_loader import ProcessedWeatherDataset
from training.base_trainer import BaseTrainer
from training.iterative_optimizer import IterativeOptimizer
from training.llm_agents import MockLLMAgent, PredictionLLM, ReflectionLLM, RefinementLLM
from training.test_pipeline import MockModel, MockPrototypeManager, MockLoss
from training.loss import TimeXLLoss
import torch.nn as nn

class DebugPredictionLLM(PredictionLLM):
    """
    Inherits from PredictionLLM to use the REAL prompt building logic,
    but mocks the generate method to print the prompt and return a fixed response.
    """
    def __init__(self):
        super().__init__(model_name="debug-mock", api_key="mock-key")

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        print("\n" + "="*60)
        print(" [DEBUG] GENERATED PROMPT SENT TO LLM ")
        print("="*60)
        print(prompt)
        print("="*60 + "\n")
        return "No Precipitation: 0.1, Rain: 0.9, Snow: 0.0"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TrainableTextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, output_dim=64, max_len=32):
        super().__init__()
        self.output_dim = output_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projector = nn.Linear(embed_dim, output_dim)
        self.vocab_size = vocab_size

    def _tokenize(self, texts):
        batch_ids = []
        for text in texts:
            words = str(text).lower().split()
            ids = [abs(hash(w)) % self.vocab_size for w in words]
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids = ids + [0] * (self.max_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

    def forward(self, texts):
        device = self.embedding.weight.device
        input_ids = self._tokenize(texts).to(device)
        x = self.embedding(input_ids)
        transformer_out = self.transformer(x)
        pooled_output = torch.mean(transformer_out, dim=1)
        projected = self.projector(pooled_output)
        return projected

class AdaptedMockModel(MockModel):
    """
    适配真实数据的 MockModel
    真实数据:
    - Time Input: [B, 1, 24] (1 channel, 24 seq_length)
    - Num Classes: ~30 (depends on weather descriptions)
    """
    def __init__(self, num_classes, k=10):
        super().__init__(num_classes=num_classes)
        # Adapt encoder to 5 channel input (from ProcessedWeatherDataset), output 256 dim
        # Input shape: [B, 24, 5] -> Flatten -> [B, 120]
        self.time_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 5, 256), 
            nn.ReLU()
        )
        
        self.text_encoder = TrainableTextEncoder(output_dim=64)
        
        # Fusion: Time(C*K) + Text(C*K) -> NumClasses
        # PM outputs [B, C*K] for both modalities.
        fusion_input_dim = num_classes * k * 2
        
        self.fusion_layer = nn.Linear(fusion_input_dim, num_classes)

    def forward(self, z_time, z_text):
        # ... (same as before) ...
        pass

    def fusion(self, sim_time, sim_text):
        # sim_time: [B, C*K], sim_text: [B, C*K]
        combined = torch.cat([sim_time, sim_text], dim=1) # [B, 2*C*K]
        return self.fusion_layer(combined)

class AdaptedMockPrototypeManager(MockPrototypeManager):
    def __init__(self, num_classes=3, k=10, time_dim=256, text_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.time_dim = time_dim
        self.text_dim = text_dim
        
        # Class-specific prototypes: [C, K, D]
        self.P_time = nn.Parameter(torch.randn(num_classes, k, time_dim))
        self.P_text = nn.Parameter(torch.randn(num_classes, k, text_dim))

    def forward(self, z_time, z_text):
        # Calculate similarity with prototypes
        # z_time: [B, D]
        # P_time: [C, K, D]
        
        # [B, C, K]
        sim_time = torch.einsum('bd,ckd->bck', z_time, self.P_time)
        sim_text = torch.einsum('bd,ckd->bck', z_text, self.P_text)
        
        # Flatten to [B, C*K]
        sim_time_flat = sim_time.reshape(z_time.size(0), -1)
        sim_text_flat = sim_text.reshape(z_text.size(0), -1)
        
        return sim_time_flat, sim_text_flat

    def fusion(self, sim_time, sim_text):
        # Not used here, handled in Model
        pass

def run_real_training_test():
    print("=== Starting Full Training Flow with Real Weather Data ===")
    
    # Processed data is in PROJECT_ROOT/data/processed_data
    # The loader expects data_dir such that file is at data_dir/processed_data/filename
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    city = 'San Francisco'
    
    # 1. 准备数据
    print(f"Loading data for {city}...")
    # Use ProcessedWeatherDataset which provides probability distribution labels
    train_dataset = ProcessedWeatherDataset(data_dir, city=city, split='train')
    val_dataset = ProcessedWeatherDataset(data_dir, city=city, split='val')
    test_dataset = ProcessedWeatherDataset(data_dir, city=city, split='test')
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.label_map)}")
    print(f"Classes: {train_dataset.label_map}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 2. 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    num_classes = len(train_dataset.label_map)
    k = 10 # Prototypes per class
    model = AdaptedMockModel(num_classes=num_classes, k=k)
    
    pm = AdaptedMockPrototypeManager(num_classes=num_classes, k=k, time_dim=256, text_dim=64) # Use adapted PM
    loss_fn = TimeXLLoss() # Use TimeXL Joint Loss
    # Include both model and prototype manager parameters in optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pm.parameters()), lr=0.001)
    
    # 3. 初始化训练器
    base_trainer = BaseTrainer(model, pm, loss_fn, optimizer, device=device)
    
    # 4. 初始化 LLM 智能体
    # We use Mock agents to avoid API costs during this integration test
    prediction_llm = DebugPredictionLLM() # Use Debug agent to verify prompt
    reflection_llm = MockLLMAgent()
    refinement_llm = MockLLMAgent()
    
    # 5. 初始化迭代优化器
    iter_optimizer = IterativeOptimizer(
        base_trainer=base_trainer,
        prediction_llm=prediction_llm,
        reflection_llm=reflection_llm,
        refinement_llm=refinement_llm,
        max_iterations=1, # Run 1 iteration for quick test
        save_dir=os.path.join(PROJECT_ROOT, 'real_data_checkpoints')
    )
    
    # 6. 开始训练
    print("\n=== Starting Optimization Loop ===")
    best_model, best_reflection, best_acc, test_acc = iter_optimizer.optimize(
        train_loader, val_loader, test_loader
    )
    
    print("\n=== Real Data Training Completed ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    if os.path.exists(os.path.join(PROJECT_ROOT, 'real_data_checkpoints', 'best_model.pth')):
        print("✅ Checkpoint file created successfully.")
        
    # 7. 原型映射 (Prototype Projection)
    print("\n=== Projecting Prototypes to Raw Data ===")
    time_expl_lib, text_expl_lib, label_expl_lib = base_trainer.project_prototypes_to_raw_data(train_loader)
    
    # Print some examples
    print("\n=== Explanation Library Examples ===")
    # Print first prototype of each class
    reverse_map = {v: k for k, v in train_dataset.label_map.items()}
    
    for c in range(num_classes):
        class_name = reverse_map.get(c, f"Class {c}")
        print(f"\n--- Class {c}: {class_name} ---")
        
        # Prototype 0 for this class
        raw_text = text_expl_lib.get((c, 0), "N/A")
        raw_time = time_expl_lib.get((c, 0), None)
        raw_label = label_expl_lib.get((c, 0), None)
        
        print(f"Prototype {c}-0 (Text Explanation): \"{raw_text}\"")
        if raw_time is not None:
             print(f"Prototype {c}-0 (Time Explanation - First 5 steps): {raw_time[:5]}")
        if raw_label is not None:
             print(f"Prototype {c}-0 (Label Explanation): {raw_label}")

    # 8. (NEW) Demonstrating LLM Prediction with Explanations
    print("\n=== Demonstrating LLM Prediction with Explanations ===")
    model.eval()
    test_iterator = iter(test_loader)
    batch = next(test_iterator)
    x_time, original_texts, targets = batch
    
    # Pick first sample
    sample_idx = 0
    sample_text = original_texts[sample_idx]
    
    # Get embeddings
    with torch.no_grad():
        x_time_gpu = x_time.to(device)
        # Slicing batch to get single item but keeping batch dim [1, ...]
        z_time = model.time_encoder(x_time_gpu[sample_idx:sample_idx+1])
        z_text = model.text_encoder([sample_text]) # encoder expects list
        
        # Get similarities [B, C*K]
        sim_time, sim_text = pm(z_time, z_text)
        
        # Average time and text similarity
        sim_combined = (sim_time + sim_text) / 2 # [1, C*K]
        
        # Get top 3 prototypes
        topk_vals, topk_indices = torch.topk(sim_combined, k=3, dim=1)
        
        retrieved_cases = []
        print(f"Test Sample Text: \"{sample_text}\"")
        print("Top 3 Retrieved Prototypes:")
        
        for i, flat_idx in enumerate(topk_indices[0]):
            flat_idx = flat_idx.item()
            # AdaptedMockPrototypeManager flattens as [C, K] -> C*K
            # So index = c * k_per_class + k_idx
            k_per_class = 10 # Defined in run_real_training_test
            c = flat_idx // k_per_class
            k_idx = flat_idx % k_per_class
            
            proto_text = text_expl_lib.get((c, k_idx), "N/A")
            proto_label = label_expl_lib.get((c, k_idx), None)
            
            # Format label string
            if proto_label is not None:
                # Assuming order 0:NoPrecip, 1:Rain, 2:Snow
                label_str = f"No Precip: {proto_label[0]:.2f}, Rain: {proto_label[1]:.2f}, Snow: {proto_label[2]:.2f}"
            else:
                label_str = "N/A"
                
            print(f"- Class {c}, Proto {k_idx}: \"{proto_text[:50]}...\" | Label: {label_str}")
            
            # Format for LLM prompt
            case_str = f"Case {i+1} Input: \"{proto_text}\"\nCase {i+1} Truth: {label_str}"
            retrieved_cases.append(case_str)
            
        # Call LLM
        task = "You are a meteorological expert. Analyze the sequence of weather descriptions recorded over the past 24 hours. Your goal is to predict the weather distribution for the NEXT 24 hours."
        # prediction_llm is defined earlier in the function scope
        prediction = prediction_llm.predict(sample_text, retrieved_cases, task_instruction=task)
        print(f"\nLLM Prediction Prompt Inputs:\nTask: {task}\n... (See llm_agents.py for full template) ...")
        print(f"LLM Prediction Result: {prediction}")

if __name__ == "__main__":
    run_real_training_test()
