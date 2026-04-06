import torch
import torch.nn as nn
from typing import List
from transformers import AutoModel, AutoTokenizer

class TimeSeriesEncoder(nn.Module):
    """
    时间序列编码器模块，用于将多变量时间序列转换为片段级特征表示。
    
    核心功能:
    - 输入: [batch_size, N, T]
    - 输出: [batch_size, T_total, h]
    - 架构: 基于CNN的一维卷积网络
    """
    def __init__(
        self, 
        input_channels: int, 
        hidden_dims: List[int], 
        kernel_sizes: List[int], 
        num_layers: int, 
        dropout_rate: float = 0.1,
        use_dropout: bool = True  # 新增参数控制Dropout
    ):
        """
        初始化 TimeSeriesEncoder
        
        Args:
            input_channels (int): 输入变量数量（通道数） N
            hidden_dims (List[int]): 各层输出维度列表 [h1, h2, ..., h_final]
            kernel_sizes (List[int]): 各层卷积核大小列表 [w1, w2, ...]
            num_layers (int): 卷积层数量
            dropout_rate (float): Dropout 比率
            use_dropout (bool): 是否使用Dropout
        """
        super(TimeSeriesEncoder, self).__init__()
        
        # 参数校验
        assert len(hidden_dims) == num_layers, "hidden_dims 长度必须等于 num_layers"
        assert len(kernel_sizes) == num_layers, "kernel_sizes 长度必须等于 num_layers"
        
        self.layers = nn.ModuleList()
        self.kernel_sizes = kernel_sizes  # 保存用于维度计算
        current_in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dims[i]
            kernel_size = kernel_sizes[i]
            
            layers_list = []
            
            # 1. 一维卷积 (使用valid padding)
            conv = nn.Conv1d(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0  # Valid padding
            )
            layers_list.append(conv)
            
            # 2. ReLU 激活
            layers_list.append(nn.ReLU())
            
            # 3. 批归一化
            layers_list.append(nn.BatchNorm1d(out_channels))
            
            # 4. Dropout (统一策略)
            if use_dropout:
                layers_list.append(nn.Dropout(dropout_rate))
            
            # 将该层的操作组合为一个 Sequential 块
            self.layers.append(nn.Sequential(*layers_list))
            
            current_in_channels = out_channels
            
        self.output_channels = hidden_dims[-1]  # 最终输出维度
        
    def compute_output_length(self, input_length: int) -> int:
        """计算给定输入长度对应的输出长度"""
        output_length = input_length
        for kernel_size in self.kernel_sizes:
            output_length = output_length - kernel_size + 1
        return output_length
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入 [batch_size, N, T]
            
        Returns:
            torch.Tensor: 输出 [batch_size, T_total, h]
        """
        # 输入: [batch_size, N, T]
        out = x
        
        # 逐层处理
        for layer in self.layers:
            out = layer(out)  # 每层都会减少时间维度: T = T - kernel_size + 1
        
        # 转置: [batch_size, h, T_total] -> [batch_size, T_total, h]
        out = out.transpose(1, 2)
        
        return out


class TextEncoder(nn.Module):
    """
    文本编码器模块，用于将原始文本转换为片段级特征表示。
    
    架构：
    - 第一级: 冻结的预训练语言模型 (PLM)，如 BERT
    - 第二级: 可训练的 CNN 编码器
    
    核心功能:
    - 输入: 文本字符串列表 (batch_size)
    - 输出: 片段级特征表示 [batch_size, L-w'+1, h']
    """
    def __init__(
        self, 
        plm_model_name: str, 
        text_hidden_dim: int, 
        text_kernel_size: int, 
        max_length: int = 512, 
        freeze_plm: bool = True, 
        dropout_rate: float = 0.1
    ):
        """
        初始化 TextEncoder
        
        Args:
            plm_model_name (str): PLM 模型名称 (如 'bert-base-uncased')
            text_hidden_dim (int): 文本隐藏维度 h'
            text_kernel_size (int): 文本卷积核大小 w'
            max_length (int): 最大文本长度
            freeze_plm (bool): 是否冻结 PLM 参数
            dropout_rate (float): Dropout 比率
        """
        super(TextEncoder, self).__init__()
        
        self.max_length = max_length
        self.text_kernel_size = text_kernel_size
        
        # 1. 初始化 PLM
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(plm_model_name)
            self.plm = AutoModel.from_pretrained(plm_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load PLM model '{plm_model_name}': {e}")
            
        # 获取 PLM 输出维度
        if hasattr(self.plm.config, 'hidden_size'):
            self.plm_output_dim = self.plm.config.hidden_size
        else:
            # 默认 BERT base 维度
            self.plm_output_dim = 768
            
        # 冻结 PLM 参数
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
                
        # 2. 构建 CNN 编码器
        # 输入: [batch_size, plm_output_dim, L]
        # 输出: [batch_size, text_hidden_dim, L-w'+1]
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.plm_output_dim,
                out_channels=text_hidden_dim,
                kernel_size=text_kernel_size
            ),
            nn.ReLU(),
            nn.BatchNorm1d(text_hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
    def get_plm_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        获取 PLM 的上下文嵌入向量
        
        Args:
            texts (List[str]): 输入文本列表
            
        Returns:
            torch.Tensor: [batch_size, L, plm_output_dim]
        """
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 将输入移至模型所在设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取嵌入
        # 如果冻结了PLM，使用no_grad上下文以节省显存并确保梯度不回传
        # 但要注意：如果是训练模式且没有完全分离，可能需要保留计算图。
        # 这里的需求是“冻结”，即 requires_grad=False。
        # 即使是 eval 模式，也可以在 forward 中运行。
        # 为了安全起见，我们直接运行模型。由于参数 requires_grad=False，梯度自然不会计算。
        
        outputs = self.plm(**inputs)
        
        # 获取 last_hidden_state: [batch_size, L, hidden_size]
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
        else:
            # 兼容其他可能的输出格式 (如 Sentence-BERT 有些变体)
            embeddings = outputs[0]
            
        return embeddings

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            texts (List[str]): 输入文本列表
            
        Returns:
            torch.Tensor: [batch_size, L-w'+1, h']
        """
        if not texts:
            raise ValueError("Input text list is empty")

        # 1. 获取 PLM 嵌入
        # embeddings: [batch_size, L, plm_output_dim]
        embeddings = self.get_plm_embeddings(texts)
        
        # 2. 调整维度以适应 Conv1d
        # Conv1d 需要输入: [batch_size, in_channels, L]
        # 当前: [batch_size, L, in_channels] -> 转置
        embeddings = embeddings.transpose(1, 2)
        
        # 3. 通过 CNN 编码器
        # cnn_out: [batch_size, text_hidden_dim, L-w'+1]
        cnn_out = self.cnn_encoder(embeddings)
        
        # 4. 调整输出维度
        # 目标输出: [batch_size, L-w'+1, text_hidden_dim]
        output = cnn_out.transpose(1, 2)
        
        return output

def test_time_series_encoder():
    """
    测试 TimeSeriesEncoder - 更新版本
    """
    print("Running TimeSeriesEncoder tests...")
    
    # 测试配置
    batch_size = 32
    N = 5       # 变量数量
    T = 100     # 时间序列长度
    
    config = {
        'input_channels': N,
        'hidden_dims': [128, 256],  # 第一层128维，第二层256维
        'kernel_sizes': [5, 1],     # 第一层kernel=5，第二层kernel=1
        'num_layers': 2,
        'dropout_rate': 0.1,
        'use_dropout': True
    }
    
    # 初始化模型
    encoder = TimeSeriesEncoder(**config)
    print("Model initialized successfully.")
    
    # 计算预期输出维度
    expected_time_steps = encoder.compute_output_length(T)
    expected_shape = (batch_size, expected_time_steps, config['hidden_dims'][-1])
    
    print(f"Input length: {T}")
    print(f"Expected output time steps: {expected_time_steps}")
    print(f"Expected output shape: {expected_shape}")
    
    # 创建模拟数据
    x = torch.randn(batch_size, N, T, requires_grad=True)
    
    # 前向传播
    output = encoder(x)
    
    # 验证输出维度
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("✅ Output dimension check passed.")
    
    # 验证梯度传播
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient not computed for input x"
    print("✅ Gradient check passed.")
    
    # 测试不同配置
    print("\nTesting different configurations...")
    
    # 测试单层网络
    single_layer_encoder = TimeSeriesEncoder(
        input_channels=5,
        hidden_dims=[256],
        kernel_sizes=[5],
        num_layers=1
    )
    single_output = single_layer_encoder(x)
    expected_single_steps = single_layer_encoder.compute_output_length(T)
    assert single_output.shape == (batch_size, expected_single_steps, 256)
    print("✅ Single layer configuration test passed.")
    
    print("TimeSeriesEncoder tests passed!")


def test_text_encoder():
    """
    测试 TextEncoder
    """
    print("\nRunning TextEncoder tests...")
    
    # 测试数据
    sample_texts = [
        "Temperature increased gradually with high humidity levels.",
        "Pressure remained stable throughout the day with light winds."
    ]
    
    # 配置
    plm_model_name = 'bert-base-uncased'
    text_hidden_dim = 64
    text_kernel_size = 3
    max_length = 128
    
    print(f"Initializing TextEncoder with {plm_model_name}...")
    # 初始化编码器
    encoder = TextEncoder(
        plm_model_name=plm_model_name,
        text_hidden_dim=text_hidden_dim,
        text_kernel_size=text_kernel_size,
        max_length=max_length,
        freeze_plm=True
    )
    
    # 1. 前向传播测试
    output = encoder(sample_texts)
    print(f"输入文本数量: {len(sample_texts)}")
    print(f"输出维度: {output.shape}")
    
    # 获取实际的 token 长度（包括 padding）
    # 由于 padding=True，所有序列长度将等于最长序列的长度，或者是 max_length（如果被截断）
    # 在这个简单的例子中，两个句子都很短，且长度相近。
    # 我们可以通过 tokenizer 检查一下预期的序列长度 L
    inputs = encoder.tokenizer(sample_texts, padding=True, return_tensors="pt")
    L = inputs['input_ids'].shape[1]
    print(f"Tokenized sequence length (L): {L}")
    
    expected_output_length = L - text_kernel_size + 1
    expected_shape = (len(sample_texts), expected_output_length, text_hidden_dim)
    
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("✅ Output dimension check passed.")
    
    # 2. 验证PLM冻结
    plm_params = [p for p in encoder.plm.parameters()]
    assert all(not p.requires_grad for p in plm_params), "PLM参数应该被冻结"
    print("✅ PLM frozen check passed.")
    
    # 3. 验证CNN可训练
    cnn_params = [p for p in encoder.cnn_encoder.parameters()]
    assert all(p.requires_grad for p in cnn_params), "CNN参数应该可训练"
    print("✅ CNN trainable check passed.")
    
    # 4. 验证梯度传播 (只对CNN部分)
    loss = output.sum()
    loss.backward()
    # 检查 CNN 第一层权重是否有梯度
    cnn_first_layer_weight = encoder.cnn_encoder[0].weight
    assert cnn_first_layer_weight.grad is not None, "CNN gradients not computed"
    print("✅ Gradient flow check passed.")
    
    print("TextEncoder tests passed!")

if __name__ == "__main__":
    test_time_series_encoder()
    test_text_encoder()
