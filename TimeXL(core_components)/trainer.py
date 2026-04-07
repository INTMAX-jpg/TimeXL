import torch
import torch.nn as nn
import copy
import os
from typing import List, Tuple, Optional, Callable, Dict
from torch.utils.data import DataLoader

class BaseTrainer:
    """
    基础训练器 - 处理单次迭代的训练
    """
    def __init__(self, model, prototype_manager, loss_fn, optimizer, device='cuda'):
        self.model = model
        self.prototype_manager = prototype_manager
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.prototype_manager.to(device)

    def train_epoch(self, dataloader: DataLoader, text_refinement_fn: Optional[Callable] = None) -> Tuple[float, float, Dict]:
        """
        单epoch训练
        
        Args:
            dataloader: 数据加载器
            text_refinement_fn: 文本优化函数（在迭代优化中使用）
            
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
            loss_details: 最后一个batch的详细损失
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        last_loss_details = {}
        
        for batch_idx, batch in enumerate(dataloader):
            # 解包批次数据
            # 假设 batch 结构为 (x_time, original_texts, targets)
            x_time, original_texts, targets = batch
            x_time = x_time.to(self.device)
            targets = targets.to(self.device)
            
            # 文本优化（如果提供）
            if text_refinement_fn is not None:
                refined_texts = text_refinement_fn(original_texts)
            else:
                refined_texts = original_texts
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            # 假设 model 有 time_encoder 和 text_encoder
            z_time = self.model.time_encoder(x_time)
            z_text = self.model.text_encoder(refined_texts)
            
            # 原型相似度计算
            sim_time, sim_text = self.prototype_manager(z_time, z_text)
            
            # 融合预测
            # 假设 model 有 fusion_layer
            # 注意：这里假设 model 是 TimeXL 主类，或者我们需要手动调用 fusion
            # 根据之前上下文，还没有 TimeXL 主类，这里先假设 model 包含了所有
            # 或者我们可以在这里直接实现简单的融合逻辑，但最好是 model 的一部分
            # 临时实现：如果 model 没有 fusion_layer，我们简单相加
            if hasattr(self.model, 'fusion_layer'):
                predictions = self.model.fusion_layer(sim_time, sim_text)
            else:
                # 简单的加权融合作为 fallback
                # sim_time: [B, num_classes * k], sim_text: [B, num_classes * k']
                # 需要先聚合到类别分数
                # 这里暂时假设 model 必须有 fusion_layer，或者我们在测试时 mock 一个
                predictions = self._simple_fusion(sim_time, sim_text)

            # 计算损失
            loss, loss_details = self.loss_fn(
                predictions, targets, z_time, z_text, self.prototype_manager
            )
            last_loss_details = loss_details
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计信息
            total_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # 打印进度 (可选，避免太多输出)
            # if batch_idx % 10 == 0:
            #     print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy, last_loss_details

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        验证集评估
        
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x_time, original_texts, targets = batch
                x_time = x_time.to(self.device)
                targets = targets.to(self.device)
                
                z_time = self.model.time_encoder(x_time)
                z_text = self.model.text_encoder(original_texts)
                
                sim_time, sim_text = self.prototype_manager(z_time, z_text)
                
                if hasattr(self.model, 'fusion_layer'):
                    predictions = self.model.fusion_layer(sim_time, sim_text)
                else:
                    predictions = self._simple_fusion(sim_time, sim_text)
                
                loss, _ = self.loss_fn(
                    predictions, targets, z_time, z_text, self.prototype_manager
                )
                
                total_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy

    def _simple_fusion(self, sim_time, sim_text):
        """
        简单的融合逻辑 fallback
        假设 sim 维度是 [B, num_classes * k]
        """
        # 这是一个临时的 fallback，实际应该在 model 中定义
        # 我们假设 num_classes = sim_time.shape[1] // k
        # 这里无法确切知道 k，只能假设。
        # 为了代码健壮性，建议用户确保 model 具有 fusion_layer
        raise NotImplementedError("Model must have a 'fusion_layer' method.")


class IterativeOptimizer:
    """
    迭代优化循环 - 实现论文的Algorithm 1
    """
    def __init__(self, base_trainer: BaseTrainer, 
                 prediction_llm, reflection_llm, refinement_llm, 
                 max_iterations: int = 5, alpha: float = 0.7,
                 save_dir: str = './checkpoints'):
        self.base_trainer = base_trainer
        self.prediction_llm = prediction_llm
        self.reflection_llm = reflection_llm
        self.refinement_llm = refinement_llm
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def optimize(self, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
        """
        执行完整迭代优化
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
        """
        # 初始化
        # s_0: 初始文本集合 (这里简化处理，假设 dataloader 中的文本就是初始文本)
        # 我们不需要显式存储所有 s_0，因为 dataloader 会提供
        # 但我们需要维护一个 current_texts_map 来支持 refine_text_context
        # 由于 dataloader 每次 yield batch，我们需要一种机制来更新文本
        # 简化起见，我们假设 refine_text_context 只是生成一个 reflection 指导，
        # 实际的文本修改可能是在 batch 处理时动态进行的，或者我们在这里维护一个全局文本库
        # 根据 Algorithm 1，是 Refine(s_t, refl_t) -> s_{t+1}
        
        # 由于我们无法直接修改 dataloader 里的数据，我们假设 text_refinement_fn 会处理
        # "如何根据当前的 reflection 修改输入文本"
        
        current_reflection = ""
        i = 0
        best_val_accuracy = 0.0
        best_encoder_state = None
        best_reflection = ""
        
        while i < self.max_iterations:
            print(f"\n=== 迭代 {i+1}/{self.max_iterations} ===")
            
            # 1. 使用当前文本训练编码器
            print("训练编码器...")
            # 定义当前的文本优化函数
            def current_refinement_fn(texts):
                # 这里调用 refinement_llm 根据 reflection 修改文本
                # 为了演示，如果 refinement_llm 是 None，则不修改
                if self.refinement_llm and current_reflection:
                     # 注意：这里可能会很慢，如果是实时调用 LLM
                     # 实际生产中可能需要缓存或批量处理
                     return [self.refine_text_context(t, current_reflection) for t in texts]
                return texts

            train_loss, train_acc, _ = self.base_trainer.train_epoch(
                train_dataloader,
                text_refinement_fn=current_refinement_fn
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # 2. 在验证集上评估
            print("验证模型...")
            val_loss, val_accuracy = self.base_trainer.validate(val_dataloader)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # 生成反思 (Reflection)
            # 这里的 reflection 应该是基于验证集上的错误案例生成的
            # 简化：我们假设 evaluate_on_validation 返回 accuracy 和 reflection
            reflection = self._generate_reflection(val_loss, val_accuracy)
            
            # 3. 检查性能提升
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_encoder_state = copy.deepcopy(self.base_trainer.model.state_dict())
                best_reflection = reflection
                print(f"✅ 新的最佳准确率: {best_val_accuracy:.4f}")
                
                # 保存最佳模型
                torch.save(best_encoder_state, os.path.join(self.save_dir, 'best_model.pth'))
            
            # 4. 更新文本和迭代计数器
            # s_{t+1} = Refine(s_t, refl)
            current_reflection = reflection
            i += 1
        
        # 最终测试
        print("\n=== 最终测试 ===")
        if best_encoder_state is not None:
            self.base_trainer.model.load_state_dict(best_encoder_state)
        
        # 测试集评估
        test_loss, test_accuracy = self.base_trainer.validate(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        return self.base_trainer.model, best_reflection, best_val_accuracy, test_accuracy

    def refine_text_context(self, text: str, reflection: str) -> str:
        """使用优化LLM优化文本"""
        if self.refinement_llm:
            # 模拟调用 LLM 生成优化后的文本
            # return self.refinement_llm.generate(f"Text: {text}\nReflection: {reflection}\nRefined:")
            return text # Placeholder
        return text

    def _generate_reflection(self, val_loss, val_acc):
        """生成反思信息"""
        if self.reflection_llm:
            # return self.reflection_llm.generate(...)
            pass
        return f"Current accuracy is {val_acc:.4f}. Focus on distinguishing similar classes."

def test_training_pipeline():
    """测试训练流水线"""
    print("Testing Training Pipeline...")
    
    # 1. Mock Components
    class MockEncoder(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            if isinstance(x, list): # Text input
                batch_size = len(x)
                return torch.randn(batch_size, 10, 64).to(next(self.parameters()).device) # [B, S, D]
            return self.linear(x) # Time input [B, N, T] -> [B, S, D] (Mocked)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_encoder = MockEncoder(100, 256) # Input T=100
            # Mock time encoder outputting [B, 20, 256] with grad
            self.time_encoder.forward = lambda x: torch.randn(x.size(0), 20, 256, requires_grad=True).to(x.device)
            
            self.text_encoder = MockEncoder(0, 64)
            # Text encoder output with grad
            self.text_encoder.forward = lambda x: torch.randn(len(x), 20, 64, requires_grad=True).to(next(self.parameters()).device)

            self.fusion_layer = nn.Linear(2, 2) # Use real layer for grad
            # Mock fusion to use inputs
            # sim_time: [B, 10], sim_text: [B, 6]
            self.fusion_layer.forward = lambda st, sx: (st[:, :2] + sx[:, :2]) # Simple combine for grad flow

            
    # Mock PrototypeManager
    class MockPrototypeManager(nn.Module):
        def __init__(self):
            super().__init__()
            self.P_time = nn.Parameter(torch.randn(10, 256))
            self.P_text = nn.Parameter(torch.randn(6, 64))
            
        def forward(self, z_time, z_text):
            B = z_time.size(0)
            # Return tensors connected to input for gradient flow
            # Simulating distances/similarities derived from z
            # z_time: [B, S, D] -> sim: [B, C*K]
            sim_time = torch.mean(z_time, dim=2)[:, :10] # Mock sim
            sim_text = torch.mean(z_text, dim=2)[:, :6]  # Mock sim
            return sim_time, sim_text
        
        def diversity_loss(self, d1, d2):
            return torch.tensor(0.1).to(self.P_time.device)

    # Mock Loss
    class MockLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, pred, target, zt, zx, pm):
            loss = F.cross_entropy(pred, target)
            return loss, {'loss': loss.item()}

    # 2. Setup
    device = 'cpu'
    model = MockModel()
    pm = MockPrototypeManager()
    loss_fn = MockLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = BaseTrainer(model, pm, loss_fn, optimizer, device)
    
    # 3. Mock Data
    # batch: (x_time, original_texts, targets)
    x_time = torch.randn(4, 5, 100)
    texts = ["text1", "text2", "text3", "text4"]
    targets = torch.randint(0, 2, (4,))
    dataloader = [(x_time, texts, targets)] # Single batch dataloader
    
    # 4. Test BaseTrainer
    print("Testing BaseTrainer...")
    avg_loss, acc, details = trainer.train_epoch(dataloader)
    print(f"Epoch Loss: {avg_loss}, Acc: {acc}")
    assert avg_loss > 0
    
    # 5. Test IterativeOptimizer
    print("Testing IterativeOptimizer...")
    optimizer_loop = IterativeOptimizer(
        trainer, None, None, None, max_iterations=2, save_dir='./test_checkpoints'
    )
    
    best_model, best_refl, best_acc, test_acc = optimizer_loop.optimize(
        dataloader, dataloader, dataloader
    )
    
    print(f"Best Val Acc: {best_acc}")
    print(f"Test Acc: {test_acc}")
    
    # Cleanup
    import shutil
    if os.path.exists('./test_checkpoints'):
        shutil.rmtree('./test_checkpoints')
        
    print("✅ Training Pipeline Tests Passed")

if __name__ == "__main__":
    import torch.nn.functional as F
    test_training_pipeline()
