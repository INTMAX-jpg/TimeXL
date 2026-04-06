
import os
import sys
import torch
import logging
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Import components
from data.synthetic_data import SyntheticTimeXLDataset
from training.base_trainer import BaseTrainer
from training.iterative_optimizer import IterativeOptimizer
from training.llm_agents import MockLLMAgent # 使用 Mock 防止消耗 Token
from training.test_pipeline import MockModel, MockPrototypeManager, MockLoss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_full_training_test():
    print("=== Starting Full Training Flow Test with Synthetic Data ===")
    
    # 1. 准备数据
    print("Generating synthetic data...")
    train_dataset = SyntheticTimeXLDataset(num_samples=64, seq_length=100, num_features=5)
    val_dataset = SyntheticTimeXLDataset(num_samples=32, seq_length=100, num_features=5)
    test_dataset = SyntheticTimeXLDataset(num_samples=32, seq_length=100, num_features=5)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # 2. 初始化模型组件
    # 注意：这里我们使用 MockModel 来快速跑通流程
    # 在真实场景中，这里应该实例化真正的 TimeXLModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = MockModel() # 这里假设 MockModel 的输入维度能匹配 Synthetic 数据
    # MockModel time_encoder input dim is 100 (matches seq_length=100 in data if transposed correctly)
    # 我们的 SyntheticData 输出 [B, N, T] (B, 5, 100)
    # MockEncoder 目前接受 [B, N, T] -> [B, S, D]
    
    pm = MockPrototypeManager()
    loss_fn = MockLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. 初始化训练器
    base_trainer = BaseTrainer(model, pm, loss_fn, optimizer, device=device)
    
    # 4. 初始化 LLM 智能体 (使用 Mock)
    prediction_llm = MockLLMAgent()
    reflection_llm = MockLLMAgent()
    refinement_llm = MockLLMAgent()
    
    # 5. 初始化迭代优化器
    iter_optimizer = IterativeOptimizer(
        base_trainer=base_trainer,
        prediction_llm=prediction_llm,
        reflection_llm=reflection_llm,
        refinement_llm=refinement_llm,
        max_iterations=2, # 测试跑两轮
        save_dir=os.path.join(PROJECT_ROOT, 'test_run_checkpoints')
    )
    
    # 6. 开始训练
    print("\n=== Starting Iterative Optimization ===")
    best_model, best_reflection, best_acc, test_acc = iter_optimizer.optimize(
        train_loader, val_loader, test_loader
    )
    
    print("\n=== Training Completed ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Reflection: {best_reflection}")
    
    # 验证输出文件
    if os.path.exists(os.path.join(PROJECT_ROOT, 'test_run_checkpoints', 'best_model.pth')):
        print("✅ Checkpoint file created successfully.")
    else:
        print("❌ Checkpoint file missing.")

if __name__ == "__main__":
    run_full_training_test()
