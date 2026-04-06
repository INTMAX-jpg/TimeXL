import os
import copy
import torch
import logging
from typing import Optional, List, Dict, Any, Callable
from torch.utils.data import DataLoader
from datetime import datetime
from .base_trainer import BaseTrainer
from .llm_agents import PredictionLLM, ReflectionLLM, RefinementLLM #尚未实现

class IterativeOptimizer:
    """
    迭代优化循环 - 实现 TimeXL 论文中的 Algorithm 1
    通过训练-验证-反思-优化循环来提升模型性能
    """
    def __init__(self, 
                 base_trainer: BaseTrainer, 
                 prediction_llm: Optional[PredictionLLM], 
                 reflection_llm: Optional[ReflectionLLM], 
                 refinement_llm: Optional[RefinementLLM], 
                 max_iterations: int = 5, 
                 alpha: float = 0.7,
                 save_dir: str = './checkpoints'):
        """
        Args:
            base_trainer: 基础训练器实例
            prediction_llm: 预测LLM智能体
            reflection_llm: 反思LLM智能体
            refinement_llm: 优化LLM智能体
            max_iterations: 最大迭代次数
            alpha: 衰减系数 (预留)
            save_dir: 检查点保存目录
        """
        self.base_trainer = base_trainer
        self.prediction_llm = prediction_llm
        self.reflection_llm = reflection_llm
        self.refinement_llm = refinement_llm
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.save_dir = save_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def optimize(self, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
        """
        执行完整迭代优化 (Algorithm 1)
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            
        Returns:
            best_model: 最佳模型
            best_reflection: 最佳反思内容
            best_val_accuracy: 最佳验证准确率
            test_accuracy: 在最佳模型上的测试准确率
        """
        current_reflection = ""
        i = 0
        best_val_accuracy = 0.0
        best_encoder_state = None
        best_reflection = ""
        
        # 缓存优化后的文本，避免重复调用LLM (Key: original_text, Value: refined_text)
        self.text_cache = {} 

        while i < self.max_iterations:
            self.logger.info(f"\n=== 迭代 {i+1}/{self.max_iterations} ===")
            
            # 1. 文本优化
            # 如果有 reflection，预先优化整个训练集的文本
            # 注意：这需要我们能够访问 dataset 里的所有文本
            # 这里我们仍然保留 text_refinement_fn 的接口，但它的实现会利用 self.text_cache
            
            # 预先计算/更新缓存 (如果需要)
            if self.refinement_llm and current_reflection:
                 self.logger.info("Optimizing texts based on reflection...")
                 # 实际上我们无法轻易遍历整个 dataset 的文本如果不 iterating
                 # 所以我们还是保持动态优化的逻辑，但是利用 cache
                 # 这里的 "efficiency" 改进主要是指避免同一文本多次 LLM 调用
                 pass

            # 定义当前的文本优化函数 (Batch-wise)
            def current_refinement_fn(texts: List[str]) -> List[str]:
                if not self.refinement_llm or not current_reflection:
                    return texts
                
                refined_texts = []
                for text in texts:
                    cache_key = f"{text}_{current_reflection}"
                    if cache_key in self.text_cache:
                        refined_texts.append(self.text_cache[cache_key])
                    else:
                        try:
                            refined = self.refinement_llm.refine(current_reflection, text)
                            self.text_cache[cache_key] = refined
                            refined_texts.append(refined)
                        except Exception as e:
                            self.logger.error(f"Text refinement failed: {e}")
                            refined_texts.append(text)
                return refined_texts

            # 2. 使用当前文本训练编码器
            self.logger.info("Training encoder...")
            train_loss, train_acc, _ = self.base_trainer.train_epoch(
                train_dataloader,
                text_refinement_fn=current_refinement_fn
            )
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # 3. 在验证集上评估
            self.logger.info("Validating model...")
            val_loss, val_accuracy = self.base_trainer.validate(val_dataloader)
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # 4. 生成反思 (Reflection)
            reflection = self._generate_reflection(val_loss, val_accuracy, val_dataloader)
            
            # 5. 检查性能提升并保存检查点
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
                best_encoder_state = copy.deepcopy(self.base_trainer.model.state_dict())
                best_reflection = reflection
                self.logger.info(f"✅ New best accuracy: {best_val_accuracy:.4f}")
            
            self.save_checkpoint(i, self.base_trainer.model, reflection, val_accuracy, is_best)
            
            # 6. 更新当前反思并增加计数器
            current_reflection = reflection
            i += 1
        
        # 最终测试
        self.logger.info("\n=== Final Testing ===")
        if best_encoder_state is not None:
            self.base_trainer.model.load_state_dict(best_encoder_state)
        
        test_loss, test_accuracy = self.base_trainer.validate(test_dataloader)
        self.logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        return self.base_trainer.model, best_reflection, best_val_accuracy, test_accuracy

    def _generate_reflection(self, val_loss: float, val_acc: float, val_dataloader: DataLoader) -> str:
        """生成反思信息"""
        if self.reflection_llm:
            try:
                status_text = f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}"
                return self.reflection_llm.reflect("High Accuracy", f"Current Accuracy: {val_acc}", status_text)
            except Exception as e:
                self.logger.error(f"Reflection generation failed: {e}")
        return f"Current accuracy is {val_acc:.4f}. Focus on distinguishing similar classes."

    def save_checkpoint(self, iteration: int, model, reflection: str, accuracy: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'reflection': reflection,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'checkpoint_iter_{iteration}.pth'
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.base_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['iteration'], checkpoint['reflection'], checkpoint['accuracy']
