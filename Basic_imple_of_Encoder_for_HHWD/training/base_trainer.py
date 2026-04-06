import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable, Dict, List
from torch.utils.data import DataLoader
import logging

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
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.model.to(device)
        self.prototype_manager.to(device)
        
        self.train_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

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
            
            # 确保模型参数在正确设备上 (Basic check on first batch)
            if batch_idx == 0:
                 assert next(self.model.parameters()).device == torch.device(self.device) or next(self.model.parameters()).device.type == torch.device(self.device).type

            # 文本优化（如果提供）
            if text_refinement_fn is not None:
                refined_texts = text_refinement_fn(original_texts)
            else:
                refined_texts = original_texts
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            z_time = self.model.time_encoder(x_time)
            z_text = self.model.text_encoder(refined_texts)
            
            # 原型相似度计算
            sim_time, sim_text = self.prototype_manager(z_time, z_text)
            
            # 融合预测
            # Check if model has a separate fusion method that takes two args, 
            # or if fusion_layer is a module (like Linear) that expects one arg.
            if hasattr(self.model, 'fusion'):
                 predictions = self.model.fusion(sim_time, sim_text)
            elif hasattr(self.model, 'fusion_layer'):
                if isinstance(self.model.fusion_layer, nn.Linear):
                     # If it's a linear layer, we likely need to concat
                     combined = torch.cat([sim_time, sim_text], dim=1)
                     predictions = self.model.fusion_layer(combined)
                else:
                     # Try calling it with two args (legacy/custom module support)
                     predictions = self.model.fusion_layer(sim_time, sim_text)
            else:
                # Fallback or error
                raise NotImplementedError("Model must have 'fusion_layer'")

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
            
            # Handle both hard labels (indices) and soft labels (probabilities)
            if targets.dim() > 1 and targets.size(1) > 1:
                # Soft labels: use argmax to get class index
                target_indices = torch.argmax(targets, dim=1)
            else:
                # Hard labels
                target_indices = targets

            correct_predictions += (predicted == target_indices).sum().item()
            total_samples += targets.size(0)
            
            # 简单日志
            if batch_idx % 100 == 0:
                self.logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # 记录历史
        self.train_history['train_loss'].append(avg_loss)
        self.train_history['train_acc'].append(accuracy)
        
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
                
                if hasattr(self.model, 'fusion'):
                    predictions = self.model.fusion(sim_time, sim_text)
                elif hasattr(self.model, 'fusion_layer'):
                    if isinstance(self.model.fusion_layer, nn.Linear):
                        combined = torch.cat([sim_time, sim_text], dim=1)
                        predictions = self.model.fusion_layer(combined)
                    else:
                        predictions = self.model.fusion_layer(sim_time, sim_text)
                else:
                    raise NotImplementedError("Model must have 'fusion_layer'")
                
                loss, _ = self.loss_fn(
                    predictions, targets, z_time, z_text, self.prototype_manager
                )
                
                total_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                
                # Handle both hard labels (indices) and soft labels (probabilities)
                if targets.dim() > 1 and targets.size(1) > 1:
                    target_indices = torch.argmax(targets, dim=1)
                else:
                    target_indices = targets
                    
                correct_predictions += (predicted == target_indices).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        self.train_history['val_loss'].append(avg_loss)
        self.train_history['val_acc'].append(accuracy)
        
        return avg_loss, accuracy

    def get_training_history(self):
        return self.train_history

    def project_prototypes_to_raw_data(self, dataloader: DataLoader) -> Tuple[Dict, Dict, Dict]:
        """
        Post-Training Step: Map abstract prototypes to nearest neighbor raw data samples.
        This creates an "Explanation Library" for interpretability.
        
        Args:
            dataloader: Training dataloader (source of raw data)
            
        Returns:
            time_expl_lib: Dict mapping (class, k) -> raw_time_series
            text_expl_lib: Dict mapping (class, k) -> raw_text_string
            label_expl_lib: Dict mapping (class, k) -> raw_label_distribution (or index)
        """
        self.logger.info("Starting Prototype Projection to Raw Data...")
        self.model.eval()
        
        # 1. Collect all embeddings and raw data
        all_z_time = []
        all_z_text = []
        all_raw_time = []
        all_raw_text = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x_time, original_texts, targets = batch
                
                # Move inputs to device for encoding
                x_time_gpu = x_time.to(self.device)
                
                # Encode
                z_time = self.model.time_encoder(x_time_gpu)
                z_text = self.model.text_encoder(original_texts)
                
                # Store (move to CPU to save GPU memory)
                all_z_time.append(z_time.cpu())
                all_z_text.append(z_text.cpu())
                all_raw_time.append(x_time.cpu()) # Keep raw time series
                all_raw_text.extend(original_texts) # Keep raw text strings
                all_labels.append(targets.cpu())
                
        # Concatenate
        all_z_time = torch.cat(all_z_time, dim=0) # [N, Dim]
        all_z_text = torch.cat(all_z_text, dim=0) # [N, Dim]
        all_raw_time = torch.cat(all_raw_time, dim=0) # [N, Seq_Len, Channels] or [N, Seq_Len]
        all_labels = torch.cat(all_labels, dim=0) # [N] or [N, C]
        
        # Keep copy of original labels (potentially soft)
        all_targets_raw = all_labels.clone()

        # Handle soft labels for indexing
        if all_labels.dim() > 1 and all_labels.size(1) > 1:
            all_labels = torch.argmax(all_labels, dim=1)

        # all_raw_text is a list of strings
        
        # 2. Get Prototypes
        # P_time: [Num_Classes, K, Dim]
        P_time = self.prototype_manager.P_time.detach().cpu()
        P_text = self.prototype_manager.P_text.detach().cpu()
        
        num_classes = P_time.size(0)
        k = P_time.size(1)
        
        time_expl_lib = {}
        text_expl_lib = {}
        label_expl_lib = {}
        
        # 3. Find Nearest Neighbors for each prototype
        for c in range(num_classes):
            # Filter samples belonging to class c
            indices_c = (all_labels == c).nonzero(as_tuple=True)[0]
            
            if len(indices_c) == 0:
                self.logger.warning(f"No samples found for class {c}. Skipping prototype projection for this class.")
                continue
                
            z_time_c = all_z_time[indices_c] # [N_c, Dim]
            z_text_c = all_z_text[indices_c] # [N_c, Dim]
            
            for j in range(k):
                # Prototype vector
                p_time = P_time[c, j].unsqueeze(0) # [1, Dim]
                p_text = P_text[c, j].unsqueeze(0) # [1, Dim]
                
                # Compute distances (Euclidean)
                # ||z - p||^2 = sum((z - p)^2)
                dists_time = torch.sum((z_time_c - p_time) ** 2, dim=1) # [N_c]
                dists_text = torch.sum((z_text_c - p_text) ** 2, dim=1) # [N_c]
                
                # Find index of min distance within class c samples
                min_idx_time_local = torch.argmin(dists_time).item()
                min_idx_text_local = torch.argmin(dists_text).item()
                
                # Map back to global index
                min_idx_time_global = indices_c[min_idx_time_local].item()
                min_idx_text_global = indices_c[min_idx_text_local].item()
                
                # Store Raw Data
                time_expl_lib[(c, j)] = all_raw_time[min_idx_time_global].numpy()
                text_expl_lib[(c, j)] = all_raw_text[min_idx_text_global]
                label_expl_lib[(c, j)] = all_targets_raw[min_idx_text_global].numpy()
                
        self.logger.info("Prototype Projection Completed.")
        return time_expl_lib, text_expl_lib, label_expl_lib
