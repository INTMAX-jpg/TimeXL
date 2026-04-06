"""
本程序是一个测试框架，Mock技术"在接口尚未完成、依赖外部系统或需要隔离测试环境时尤为有用"：

依赖未完成：在实际开发中，LLM组件可能尚未完成，但需要测试训练管道的其他部分
测试环境简化：不需要依赖真实外部服务或复杂数据环境
测试可控性：可以精确控制输入和期望输出，验证特定场景
提高测试效率：避免了等待真实LLM响应或处理复杂数据的耗时
"""
"""
在调用实际API之前，我们采用MockLLMAgent，它是一个模拟LLM的类，不会调用真实的LLM API，而是返回预设的固定响应。
这些返回值是完全设定好的，不是随机的，也不是基于输入动态生成的。从代码中可以看到：
predict方法总是返回 "Up"
reflect方法总是返回 "The text missed key volatility indicators."
refine方法返回原始文本加上固定后缀，例如："The stock market has been volatile recently due to economic uncertainty. [Refined based on: The text missed key volatility indicators.]"
generate方法总是返回 "Mock generation"
这些预设的返回值在测试中是确定的，这正是Mock测试的特性：我们通过预设已知的输入/输出来验证流程是否正确，而不是依赖真实的、可能变化的LLM响应。
在测试中，我们主要验证的是流程是否正确，而不是LLM的预测准确性。例如，我们验证：
PredictionLLM是否被正确调用
ReflectionLLM是否能生成反思
RefinementLLM是否能优化文本
这些组件是否按预期顺序工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import unittest
import logging
import gc
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset

# Import refactored modules #重构模型（就是我们自己编写的）
from training.base_trainer import BaseTrainer
from training.iterative_optimizer import IterativeOptimizer
from training.llm_agents import MockLLMAgent, PredictionLLM, ReflectionLLM, RefinementLLM
from training.config import TrainingConfig

# 配置日志
logging.basicConfig(level=logging.INFO)

class MockEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        if input_dim > 0:
            self.layer = nn.Linear(input_dim, output_dim)
            self.is_text = False
        else:
            self.layer = nn.Embedding(10, output_dim) # Assume vocab size 10
            self.is_text = True
    
    def forward(self, x):
        if self.is_text:
            # x is list of strings. Map to dummy indices.
            # Simple deterministic mapping
            indices = torch.tensor([len(s) % 10 for s in x], device=self.layer.weight.device)
            return self.layer(indices).unsqueeze(1) # [B, 1, D]
        else:
            # x is [B, N, input_dim]
            return self.layer(x)

class MockModel(nn.Module):
    def __init__(self, num_classes=3): # Changed default to 3 to match synthetic data
        super().__init__()
        # Mock components
        self.time_encoder = MockEncoder(100, 256)
        self.text_encoder = MockEncoder(0, 64)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(4, num_classes) 

    def fusion(self, sim_time, sim_text):
         # 使用真实的融合层 
         combined = torch.cat([sim_time, sim_text], dim=1)  # [B, 4] 
         return self.fusion_layer(combined)  # [B, num_classes]

    def forward(self, z_time, z_text):
         # 模拟相似度计算 - 这里我们假设 PrototypeManager 已经计算了相似度
         # 但在 BaseTrainer 中，先 encode 得到 z，然后 PM 得到 sim，然后 fusion 得到 pred
         # 这个 forward 方法可能不会被直接调用，而是组件被分别调用
         # 但为了完整性，我们可以保留它
         
         # 如果被直接调用，我们需要自己计算 sim (简化版)
         sim_time = torch.mean(z_time, dim=1)[:, :2] 
         sim_text = torch.mean(z_text, dim=1)[:, :2] 
         return self.fusion(sim_time, sim_text)

    def to(self, device):
        super().to(device)
        return self

class MockPrototypeManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.P_time = nn.Parameter(torch.randn(10, 256))
        self.P_text = nn.Parameter(torch.randn(6, 64))
        
    def forward(self, z_time, z_text):
        # Use parameters to ensure gradients flow
        # z_time: [B, S, 256]
        # P_time: [10, 256]
        
        z_time_mean = torch.mean(z_time, dim=1) # [B, 256]
        z_text_mean = torch.mean(z_text, dim=1) # [B, 64]
        
        # Matrix multiplication to get similarity scores
        sim_time = torch.matmul(z_time_mean, self.P_time.T) # [B, 10]
        sim_text = torch.matmul(z_text_mean, self.P_text.T) # [B, 6]
        
        return sim_time[:, :2], sim_text[:, :2]
    
    def diversity_loss(self, d1, d2):
        return torch.tensor(0.1, requires_grad=True)

class MockLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, zt, zx, pm):
        loss = F.cross_entropy(pred, target)
        return loss, {'loss': loss.item()}

class TestTrainingPipeline(unittest.TestCase):
    
    def setUp(self):
        # Setup directories
        self.test_dir = "./test_checkpoints"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
        # Setup components
        self.device = 'cpu'
        self.model = MockModel()
        self.pm = MockPrototypeManager()
        self.loss_fn = MockLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.base_trainer = BaseTrainer(
            self.model, self.pm, self.loss_fn, self.optimizer, self.device
        )
        
        # Create Mock Data
        self.batch_size = 4
        x_time = torch.randn(self.batch_size, 5, 100)
        texts = ["text1", "text2", "text3", "text4"]
        targets = torch.randint(0, 2, (self.batch_size,))
        
        # Custom Collate for mixed data types
        def collate_fn(batch):
            x_t = torch.stack([item[0] for item in batch])
            txt = [item[1] for item in batch]
            tgt = torch.stack([item[2] for item in batch])
            return x_t, txt, tgt

        dataset = []
        for i in range(self.batch_size * 2):
            dataset.append((x_time[0], texts[0], targets[0]))
            
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        
        # Mock LLMs
        self.mock_llm = MockLLMAgent()

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_base_trainer_train_epoch(self):
        print("\nTesting BaseTrainer.train_epoch...")
        avg_loss, acc, details = self.base_trainer.train_epoch(self.dataloader)
        print(f"Epoch Loss: {avg_loss}, Acc: {acc}")
        self.assertGreater(avg_loss, 0)
        self.assertIsInstance(details, dict)
        
        # 检查训练历史
        history = self.base_trainer.get_training_history()
        self.assertGreater(len(history['train_loss']), 0)

    def test_base_trainer_validate(self):
        print("\nTesting BaseTrainer.validate...")
        avg_loss, acc = self.base_trainer.validate(self.dataloader)
        print(f"Val Loss: {avg_loss}, Acc: {acc}")
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_iterative_optimizer_full_loop(self):
        print("\nTesting IterativeOptimizer full loop...")
        
        # Initialize Optimizer with Mock LLMs
        optimizer_loop = IterativeOptimizer(
            base_trainer=self.base_trainer,
            prediction_llm=self.mock_llm, # Reusing mock for all for simplicity
            reflection_llm=self.mock_llm,
            refinement_llm=self.mock_llm,
            max_iterations=2,
            save_dir=self.test_dir
        )
        
        best_model, best_refl, best_acc, test_acc = optimizer_loop.optimize(
            self.dataloader, self.dataloader, self.dataloader
        )
        
        print(f"Best Val Acc: {best_acc}")
        print(f"Test Acc: {test_acc}")
        print(f"Best Reflection: {best_refl}")
        
        # Verify LLM calls
        print(f"LLM Call History: {self.mock_llm.call_history}")
        self.assertTrue(len(self.mock_llm.call_history) > 0, "LLM should be called")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'best_model.pth')))
        
        # 检查迭代检查点
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'checkpoint_iter_0.pth')))

    def test_gradient_flow(self):
        """验证梯度是否正确流动"""
        print("\nTesting gradient flow...")
        
        # 单批次训练 
        avg_loss, acc, details = self.base_trainer.train_epoch(self.dataloader)
        
        # 检查关键参数的梯度 
        model_params = list(self.model.parameters())
        prototype_params = list(self.pm.parameters())
        
        # 至少有一些参数应该有梯度 
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_params) 
        self.assertTrue(has_gradients, "模型参数应该有梯度") 
        
        has_prototype_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in prototype_params) 
        self.assertTrue(has_prototype_gradients, "原型参数应该有梯度") 
        
        print("✅ 梯度流动测试通过")

    def test_memory_usage(self):
        """测试内存使用情况"""
        print("\nTesting memory usage...")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 
        gc.collect() 
        
        # 记录初始内存 
        if torch.cuda.is_available(): 
            initial_memory = torch.cuda.memory_allocated() 
        
        # 执行训练 
        self.base_trainer.train_epoch(self.dataloader)
        
        # 检查内存释放 
        if torch.cuda.is_available(): 
            final_memory = torch.cuda.memory_allocated() 
            memory_increase = final_memory - initial_memory 
            print(f"内存增加: {memory_increase / 1024**2:.2f} MB") 
            
            # 确保没有内存泄漏 
            self.assertLess(memory_increase, 500 * 1024**2, "内存增加不应超过500MB") 
        
        print("✅ 内存使用测试通过")

if __name__ == '__main__':
    unittest.main()
