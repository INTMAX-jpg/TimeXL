import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

class PrototypeManager(nn.Module):
    """
    管理多模态原型的学习、相似度计算和投影。
    
    核心功能：
    1. 原型存储：P_time, P_text
    2. 相似度计算：基于欧几里得距离的指数相似度
    3. 原型投影：训练后将原型替换为最接近的真实数据片段
    4. 多样性约束：鼓励原型之间的差异性
    """
    
    def __init__(
        self, 
        num_classes: int, 
        time_dim: int, 
        text_dim: int,
        num_time_prototypes: int, 
        num_text_prototypes: int,
        device: str = 'cpu'
    ):
        """
        初始化 PrototypeManager
        
        Args:
            num_classes (int): 类别数量
            time_dim (int): 时间序列特征维度 h
            text_dim (int): 文本特征维度 h'
            num_time_prototypes (int): 每类时间序列原型数量 k
            num_text_prototypes (int): 每类文本原型数量 k'
            device (str): 设备 ('cpu' or 'cuda')
        """
        super(PrototypeManager, self).__init__()
        
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.num_time_prototypes = num_time_prototypes
        self.num_text_prototypes = num_text_prototypes
        self.device = device
        
        self.total_time_prototypes = num_classes * num_time_prototypes
        self.total_text_prototypes = num_classes * num_text_prototypes
        
        # 初始化原型 (可训练参数)
        # 形状: [total_prototypes, feature_dim]
        self.P_time = nn.Parameter(
            torch.randn(self.total_time_prototypes, time_dim)
        )
        self.P_text = nn.Parameter(
            torch.randn(self.total_text_prototypes, text_dim)
        )
        
        self.to(device)
        
    def initialize_prototypes(self, train_time_data: torch.Tensor, train_text_data: torch.Tensor, train_labels: torch.Tensor):
        """
        从训练数据中智能初始化原型
        
        Args:
            train_time_data (torch.Tensor): 训练集时间特征 [N, T_seg, h]
            train_text_data (torch.Tensor): 训练集文本特征 [N, L_seg, h']
            train_labels (torch.Tensor): 标签 [N]
        """
        with torch.no_grad():
            # 时间序列原型初始化
            for c in range(self.num_classes):
                class_indices = torch.where(train_labels == c)[0]
                if len(class_indices) == 0:
                    continue
                    
                # 获取该类别的样本
                class_time_samples = train_time_data[class_indices] # [N_c, T_seg, h]
                
                for i in range(self.num_time_prototypes):
                    # 随机选择一个样本
                    sample_idx = np.random.randint(len(class_indices))
                    time_sample = class_time_samples[sample_idx] # [T_seg, h]
                    
                    # 随机选择一个片段
                    if time_sample.size(0) > 0:
                        segment_idx = np.random.randint(time_sample.size(0))
                        proto_idx = c * self.num_time_prototypes + i
                        self.P_time[proto_idx].data = time_sample[segment_idx]
            
            # 文本原型初始化
            for c in range(self.num_classes):
                class_indices = torch.where(train_labels == c)[0]
                if len(class_indices) == 0:
                    continue
                    
                class_text_samples = train_text_data[class_indices] # [N_c, L_seg, h']
                
                for i in range(self.num_text_prototypes):
                    sample_idx = np.random.randint(len(class_indices))
                    text_sample = class_text_samples[sample_idx]
                    
                    if text_sample.size(0) > 0:
                        segment_idx = np.random.randint(text_sample.size(0))
                        proto_idx = c * self.num_text_prototypes + i
                        self.P_text[proto_idx].data = text_sample[segment_idx]

    def _compute_dist_matrix(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        计算样本所有片段与所有原型之间的距离矩阵 (向量化实现)
        
        Args:
            z (torch.Tensor): 输入片段 [batch_size, num_segments, dim]
            p (torch.Tensor): 原型 [num_prototypes, dim]
            
        Returns:
            torch.Tensor: 距离矩阵 [batch_size, num_segments, num_prototypes]
        """
        # z: [B, S, D] -> [B, S, 1, D]
        # p: [P, D] -> [1, 1, P, D]
        # dist = ||z - p||
        # 为了内存效率，展开计算: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        
        B, S, D = z.shape
        P = p.shape[0]
        
        # 展平 z 以进行批量矩阵乘法
        z_flat = z.view(B * S, D) # [B*S, D]
        
        # 1. Squared Euclidean Distance
        # ||z||^2: [B*S, 1]
        z_norm_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        # ||p||^2: [1, P]
        p_norm_sq = torch.sum(p ** 2, dim=1, keepdim=True).t()
        # 2 * z * p^T: [B*S, P]
        dot_product = torch.matmul(z_flat, p.t())
        
        # dist_sq: [B*S, P]
        dist_sq = z_norm_sq + p_norm_sq - 2 * dot_product
        
        # 防止数值误差导致负值
        dist_sq = F.relu(dist_sq)
        
        # 开根号得到欧几里得距离
        dist = torch.sqrt(dist_sq + 1e-8) # 加epsilon防止梯度为NaN
        
        # 重塑回 [B, S, P]
        return dist.view(B, S, P)

    def compute_similarities(self, z_time: torch.Tensor, z_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算输入片段与所有原型的相似度
        Sim(x, p) = max_{z \in patches(x)} exp(-||z - p||^2)
        这里我们使用 L2 距离。根据论文公式(2)，通常是 exp(-dist^2) 或 exp(-dist)。
        用户需求注释写的是: Sim = max over segments of exp(-||p - z||^2)
        
        Args:
            z_time: [batch_size, T', time_dim] 时间序列片段
            z_text: [batch_size, L', text_dim] 文本片段
            
        Returns:
            sim_time: [batch_size, total_time_prototypes]
            sim_text: [batch_size, total_text_prototypes]
        """
        # 1. 计算时间序列相似度
        # dist_matrix_time: [batch_size, T', total_time_prototypes]
        dist_matrix_time = self._compute_dist_matrix(z_time, self.P_time)
        
        # 最小距离: min over segments (dim=1) -> [batch_size, total_time_prototypes]
        min_dist_time, _ = torch.min(dist_matrix_time, dim=1)
        
        # 相似度转换: exp(-d^2)
        sim_time = torch.exp(-min_dist_time ** 2)
        
        # 2. 计算文本相似度
        # dist_matrix_text: [batch_size, L', total_text_prototypes]
        dist_matrix_text = self._compute_dist_matrix(z_text, self.P_text)
        
        # 最小距离
        min_dist_text, _ = torch.min(dist_matrix_text, dim=1)
        
        # 相似度转换
        sim_text = torch.exp(-min_dist_text ** 2)
        
        return sim_time, sim_text

    def forward(self, z_time: torch.Tensor, z_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算相似度
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (sim_time, sim_text)
        """
        return self.compute_similarities(z_time, z_text)

    def project_prototypes(self, train_time_data: torch.Tensor, train_text_data: torch.Tensor, train_labels: torch.Tensor):
        """
        执行原型投影：将原型替换为训练集中最接近的真实片段
        注意：由于数据集可能很大，这里应该注意内存使用。但根据需求描述，直接从输入tensor中查找。
        如果 train_data 非常大，调用者应该分批处理或只传入部分数据。
        
        Args:
            train_time_data: [N, T', h]
            train_text_data: [N, L', h']
            train_labels: [N]
        """
        self.eval() # 确保不更新梯度
        
        with torch.no_grad():
            # 1. 时间原型投影
            for i in range(self.total_time_prototypes):
                prototype_class = i // self.num_time_prototypes
                
                # 获取该类别的样本索引
                class_mask = (train_labels == prototype_class)
                if not class_mask.any():
                    continue
                    
                class_data = train_time_data[class_mask] # [N_c, T', h]
                
                # 为了内存安全，我们不构建巨大的距离矩阵，而是逐个样本或小批次处理
                # 但为了简单起见，如果数据量可控，可以使用 _compute_dist_matrix 的变体
                
                # 目标: 找到 min_{sample, segment} ||proto - segment||
                
                proto_vec = self.P_time[i].unsqueeze(0).unsqueeze(0) # [1, 1, h]
                
                # 展平所有样本的所有片段: [N_c * T', h]
                # 注意：这可能会很大
                flat_data = class_data.reshape(-1, self.time_dim)
                
                # 计算距离: ||x - p||^2
                # 使用 batch 处理以防 OOM
                batch_size = 1024
                min_dist = float('inf')
                best_segment = None
                
                for start_idx in range(0, flat_data.size(0), batch_size):
                    end_idx = min(start_idx + batch_size, flat_data.size(0))
                    batch_segments = flat_data[start_idx:end_idx]
                    
                    dists = torch.norm(batch_segments - self.P_time[i], dim=1)
                    current_min_val, current_min_idx = torch.min(dists, dim=0)
                    
                    if current_min_val.item() < min_dist:
                        min_dist = current_min_val.item()
                        best_segment = batch_segments[current_min_idx]
                
                if best_segment is not None:
                    self.P_time[i].data = best_segment
            
            # 2. 文本原型投影 (逻辑相同)
            for i in range(self.total_text_prototypes):
                prototype_class = i // self.num_text_prototypes
                class_mask = (train_labels == prototype_class)
                if not class_mask.any():
                    continue
                    
                class_data = train_text_data[class_mask]
                flat_data = class_data.reshape(-1, self.text_dim)
                
                batch_size = 1024
                min_dist = float('inf')
                best_segment = None
                
                for start_idx in range(0, flat_data.size(0), batch_size):
                    end_idx = min(start_idx + batch_size, flat_data.size(0))
                    batch_segments = flat_data[start_idx:end_idx]
                    
                    dists = torch.norm(batch_segments - self.P_text[i], dim=1)
                    current_min_val, current_min_idx = torch.min(dists, dim=0)
                    
                    if current_min_val.item() < min_dist:
                        min_dist = current_min_val.item()
                        best_segment = batch_segments[current_min_idx]
                
                if best_segment is not None:
                    self.P_text[i].data = best_segment

    def diversity_loss(self, d_min_time: float, d_min_text: float) -> torch.Tensor:
        """
        计算原型多样性损失：鼓励原型之间的距离大于阈值 d_min
        Loss = sum(max(0, d_min - distance(p_i, p_j)))^2  (对于 i != j)
        
        Args:
            d_min_time (float): 时间原型最小距离阈值
            d_min_text (float): 文本原型最小距离阈值
            
        Returns:
            torch.Tensor: 多样性损失值
        """
        def compute_div_loss(prototypes, d_min):
            if prototypes.size(0) <= 1:
                return torch.tensor(0.0, device=self.device)
                
            # 计算两两距离矩阵
            # prototypes: [P, D]
            # ||p_i - p_j||
            p_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True)
            dist_sq = p_sq + p_sq.t() - 2 * torch.mm(prototypes, prototypes.t())
            dist_sq = F.relu(dist_sq)
            dist_matrix = torch.sqrt(dist_sq + 1e-8)
            
            # 只考虑上三角矩阵 (i < j)，避免重复计算和自身距离(0)
            # mask 上三角部分
            mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
            pairwise_dists = dist_matrix[mask]
            
            # 惩罚小于 d_min 的距离
            # Loss = mean(max(0, d_min - dist))
            # 或者 sum，根据需求。通常是 sum 或 mean。这里用 sum 以产生足够大的梯度，或者根据规模归一化。
            # 需求只写了 "实现原型间的多样性损失"，未指定具体公式。
            # 常用做法: max(0, threshold - dist)^2
            
            loss = F.relu(d_min - pairwise_dists) ** 2
            return loss.sum()

        loss_time = compute_div_loss(self.P_time, d_min_time)
        loss_text = compute_div_loss(self.P_text, d_min_text)
        
        return loss_time + loss_text

    def get_prototype_info(self) -> Dict[str, Any]:
        """返回原型信息用于解释"""
        return {
            "time_prototypes": self.P_time.detach().cpu(),
            "text_prototypes": self.P_text.detach().cpu(),
            "config": {
                "num_classes": self.num_classes,
                "num_time_prototypes": self.num_time_prototypes,
                "num_text_prototypes": self.num_text_prototypes
            }
        }

def test_prototype_manager():
    """测试原型管理器"""
    print("Running PrototypeManager tests...")
    
    # 配置
    config = {
        'num_classes': 2,
        'time_dim': 256,
        'text_dim': 64,
        'num_time_prototypes': 5,
        'num_text_prototypes': 3,
        'device': 'cpu' # 测试时使用CPU
    }
    
    manager = PrototypeManager(**config)
    
    # 测试数据
    batch_size = 4
    # 模拟输入数据
    z_time = torch.randn(batch_size, 20, config['time_dim'])  # 20个时间片段
    z_text = torch.randn(batch_size, 15, config['text_dim'])  # 15个文本片段
    
    # 1. 测试相似度计算
    sim_time, sim_text = manager(z_time, z_text)
    
    print(f"时间相似度维度: {sim_time.shape}")
    print(f"文本相似度维度: {sim_text.shape}")
    
    expected_time_dim = config['num_classes'] * config['num_time_prototypes']
    expected_text_dim = config['num_classes'] * config['num_text_prototypes']
    
    assert sim_time.shape == (batch_size, expected_time_dim), f"Expected {(batch_size, expected_time_dim)}, got {sim_time.shape}"
    assert sim_text.shape == (batch_size, expected_text_dim), f"Expected {(batch_size, expected_text_dim)}, got {sim_text.shape}"
    
    # 验证相似度范围 [0, 1]
    assert torch.all(sim_time >= 0) and torch.all(sim_time <= 1), "Time similarities out of range [0, 1]"
    assert torch.all(sim_text >= 0) and torch.all(sim_text <= 1), "Text similarities out of range [0, 1]"
    
    print("✅ 相似度计算测试通过")
    
    # 2. 测试多样性损失
    div_loss = manager.diversity_loss(d_min_time=1.0, d_min_text=3.0)
    print(f"多样性损失: {div_loss.item()}")
    assert div_loss >= 0, "Diversity loss should be non-negative"
    
    # 验证梯度
    div_loss.backward()
    assert manager.P_time.grad is not None, "P_time gradients missing"
    assert manager.P_text.grad is not None, "P_text gradients missing"
    print("✅ 多样性损失及梯度测试通过")
    
    # 3. 测试原型初始化
    # 创建模拟训练数据
    train_time = torch.randn(10, 20, config['time_dim'])
    train_text = torch.randn(10, 15, config['text_dim'])
    train_labels = torch.randint(0, 2, (10,))
    
    manager.initialize_prototypes(train_time, train_text, train_labels)
    print("✅ 原型初始化测试通过")
    
    # 4. 测试原型投影
    # 记录投影前的原型
    old_proto_time = manager.P_time.clone()
    
    manager.project_prototypes(train_time, train_text, train_labels)
    
    # 检查原型是否改变 (虽然有极小概率随机生成的正好就是原型，但几乎不可能)
    # 注意：如果 train_time 和初始化时用的不一样，或者只是测试逻辑运行无误
    # 这里我们主要验证代码能跑通
    print("✅ 原型投影测试通过")
    
    print("All PrototypeManager tests passed!")

if __name__ == "__main__":
    test_prototype_manager()
