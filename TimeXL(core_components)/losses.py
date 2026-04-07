import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
try:
    from .prototypes import PrototypeManager
except ImportError:
    from prototypes import PrototypeManager

class TimeXLLoss(nn.Module):
    """
    TimeXL完整损失函数
    L = L_CE + λ1 * L_c + λ2 * L_e + λ3 * L_d
    
    其中：
    - L_CE: 交叉熵损失（主分类损失）
    - L_c: 聚类损失（数据片段靠近原型）
    - L_e: 证据损失（原型靠近数据片段） 
    - L_d: 多样性损失（原型间距离约束）
    """
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1, lambda3: float = 0.1, 
                 d_min_time: float = 1.0, d_min_text: float = 3.0, reduction: str = 'mean'):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.d_min_time = d_min_time
        self.d_min_text = d_min_text
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

    def get_loss_weights(self) -> Dict[str, float]:
        """返回当前损失权重配置"""
        return {
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'd_min_time': self.d_min_time,
            'd_min_text': self.d_min_text
        }

    def clustering_loss_vectorized(self, z_time: torch.Tensor, z_text: torch.Tensor, 
                                 P_time: torch.Tensor, P_text: torch.Tensor) -> torch.Tensor:
        """
        向量化版本的聚类损失计算（更高效）
        L_c = Σ min_p ||z_j - p_i||²
        
        Args:
            z_time: [B, S, D]
            z_text: [B, S', D']
            P_time: [P, D]
            P_text: [P', D']
        """
        def compute_single_modal_clustering(z, p):
            # z: [B, S, D], p: [P, D]
            B, S, D = z.shape
            
            # 重塑为 [B*S, D] 并与所有原型计算距离
            # detach P to stop gradient
            z_flat = z.reshape(B * S, D)
            p_detached = p.detach()
            
            # cdist 计算的是欧几里得距离 (L2 norm)
            # distances: [B*S, P]
            distances = torch.cdist(z_flat, p_detached)
            
            # 每个片段到最近原型的距离
            min_distances, _ = torch.min(distances, dim=1)
            
            # 损失为平方距离的均值
            return torch.mean(min_distances ** 2)

        loss_time = compute_single_modal_clustering(z_time, P_time)
        loss_text = compute_single_modal_clustering(z_text, P_text)
        
        return loss_time + loss_text

    def evidence_loss_vectorized(self, z_time: torch.Tensor, z_text: torch.Tensor, 
                               P_time: torch.Tensor, P_text: torch.Tensor) -> torch.Tensor:
        """
        向量化版本的证据损失计算
        L_e = Σ min_z ||p_i - z_j||²
        
        Args:
            z_time: [B, S, D]
            z_text: [B, S', D']
            P_time: [P, D]
            P_text: [P', D']
        """
        def compute_single_modal_evidence(z, p):
            # z: [B, S, D], p: [P, D]
            B, S, D = z.shape
            
            # 重塑为 [B*S, D]
            # detach z to stop gradient
            z_flat = z.reshape(B * S, D).detach()
            
            # distances: [P, B*S]
            distances = torch.cdist(p, z_flat)
            
            # 每个原型到最近数据片段的距离
            min_distances, _ = torch.min(distances, dim=1)
            
            # 损失为平方距离的均值
            return torch.mean(min_distances ** 2)

        loss_time = compute_single_modal_evidence(z_time, P_time)
        loss_text = compute_single_modal_evidence(z_text, P_text)
        
        return loss_time + loss_text

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                z_time: torch.Tensor, z_text: torch.Tensor, 
                prototype_manager: PrototypeManager) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算完整损失
        
        Args:
            predictions: 模型预测 [batch_size, num_classes]
            targets: 真实标签 [batch_size] 
            z_time: 时间序列片段 [batch_size, num_segments, time_dim]
            z_text: 文本片段 [batch_size, num_segments, text_dim]
            prototype_manager: 原型管理器实例
            
        Returns:
            total_loss: 总损失
            loss_details: 详细损失信息
        """
        # 1. 交叉熵损失
        ce_loss = self.ce_loss(predictions, targets)
        
        # 2. 聚类损失
        cluster_loss = self.clustering_loss_vectorized(
            z_time, z_text,
            prototype_manager.P_time,
            prototype_manager.P_text
        )
        
        # 3. 证据损失 
        evidence_loss = self.evidence_loss_vectorized(
            z_time, z_text,
            prototype_manager.P_time,
            prototype_manager.P_text
        )
        
        # 4. 多样性损失（直接从原型管理器获取）
        diversity_loss = prototype_manager.diversity_loss(
            self.d_min_time, self.d_min_text
        )
        
        # 5. 组合所有损失
        total_loss = (ce_loss + 
                     self.lambda1 * cluster_loss + 
                     self.lambda2 * evidence_loss + 
                     self.lambda3 * diversity_loss)
        
        # 返回详细损失信息用于监控
        loss_details = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'cluster_loss': cluster_loss,
            'evidence_loss': evidence_loss,
            'diversity_loss': diversity_loss
        }
        
        return total_loss, loss_details

def test_timexl_loss():
    """测试TimeXL损失函数"""
    print("Testing TimeXLLoss...")
    
    # 配置
    loss_fn = TimeXLLoss(
        lambda1=0.1, lambda2=0.1, lambda3=0.1,
        d_min_time=1.0, d_min_text=3.0
    )
    
    # 模拟数据
    batch_size = 4
    num_classes = 2
    time_dim, text_dim = 256, 64
    num_segments = 20
    
    # 预测和真实值
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 片段表示 (需要梯度)
    z_time = torch.randn(batch_size, num_segments, time_dim, requires_grad=True)
    z_text = torch.randn(batch_size, num_segments, text_dim, requires_grad=True)
    
    # 原型管理器
    prototype_manager = PrototypeManager(
        num_classes=num_classes,
        time_dim=time_dim,
        text_dim=text_dim,
        num_time_prototypes=5,
        num_text_prototypes=3
    )
    
    # 计算损失
    total_loss, loss_details = loss_fn(
        predictions, targets, z_time, z_text, prototype_manager
    )
    
    print("损失详情:")
    for name, value in loss_details.items():
        print(f"  {name}: {value.item():.4f}")
    
    # 验证损失值合理性
    assert total_loss.item() > 0, "总损失应为正数"
    assert all(v.item() >= 0 for v in loss_details.values()), "所有损失分量应为非负"
    
    # 验证梯度
    total_loss.backward()
    
    # 检查梯度是否存在
    # L_c 应该给 z 梯度
    # L_e 应该给 P 梯度
    # L_d 应该给 P 梯度
    assert z_time.grad is not None, "z_time gradient missing"
    assert z_text.grad is not None, "z_text gradient missing"
    assert prototype_manager.P_time.grad is not None, "P_time gradient missing"
    assert prototype_manager.P_text.grad is not None, "P_text gradient missing"
    
    print("✅ 梯度计算正常")
    print("所有损失函数测试通过!")

if __name__ == "__main__":
    test_timexl_loss()
