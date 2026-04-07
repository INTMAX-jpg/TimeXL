import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

class TrainableTextEncoder(nn.Module):
    """
    可训练的文本编码器
    功能：将天气描述文本转换为固定维度的特征向量
    """
    def __init__(self, vocab_size=1000, embed_dim=128, output_dim=64, max_len=32):
        # 继承nn.Module父类初始化
        super().__init__()
        # 编码器最终输出的特征维度
        self.output_dim = output_dim
        # 文本分词后的最大长度，超出截断，不足填充
        self.max_len = max_len
        # 词嵌入层：将词汇索引转换为稠密向量，vocab_size词汇量，embed_dim嵌入维度
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 定义Transformer编码器层，d_model特征维度，nhead注意力头数，batch_first批次维度在前
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        # 堆叠2层Transformer编码器层，提取文本语义特征
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # 线性投影层：将Transformer输出的特征映射到指定输出维度
        self.projector = nn.Linear(embed_dim, output_dim)
        # 词汇表大小，用于哈希取模限制索引范围
        self.vocab_size = vocab_size

    def _stable_hash(self, s):
        """
        稳定哈希函数：保证相同字符串每次运行生成相同哈希值
        这样不用预定义词典，直接哈希编码，非常轻量
        :param s: 输入字符串
        :return: 哈希后的整数
        """
        # 对字符串进行utf8编码，通过md5哈希，转换为16进制整数
        return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

    def _tokenize(self, texts):
        """
        文本分词与索引化，将文本批量转换为模型可接收的索引序列
        :param texts: 文本批次列表
        :return: 张量格式的索引序列 [batch_size, max_len]
        """
        batch_ids = []
        # 遍历批次内每一条文本
        for text in texts:
            # 转换为字符串、小写，按空格分词
            words = str(text).lower().split()
            # 对每个单词哈希取模，映射到词汇表范围内的索引
            ids = [self._stable_hash(w) % self.vocab_size for w in words]
            # 文本长度超过最大长度则截断,保证每个文本的长度都相同
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            # 长度不足则用0填充至最大长度
            else:
                ids = ids + [0] * (self.max_len - len(ids))
            batch_ids.append(ids)
        # 转换为长整型张量，适配嵌入层输入类型
        return torch.tensor(batch_ids, dtype=torch.long)

    def forward(self, texts):
        """
        前向传播：文本编码主逻辑
        :param texts: 输入文本批次
        :return: 文本特征向量 [batch_size, output_dim]
        """
        # 获取模型所在设备（cpu/cuda）
        device = self.embedding.weight.device
        # 文本分词索引化，并迁移至模型对应设备
        input_ids = self._tokenize(texts).to(device)
        # 词嵌入：[B, max_len] -> [B, max_len, embed_dim]
        x = self.embedding(input_ids)
        # Transformer编码提取语义特征，维度保持不变
        transformer_out = self.transformer(x)
        # 均值池化：对序列维度求平均，将句子压缩为单个向量 [B, max_len, embed_dim] -> [B, embed_dim]
        pooled_output = torch.mean(transformer_out, dim=1)
        # 线性投影，映射到目标输出维度
        projected = self.projector(pooled_output)
        # 返回最终文本特征
        return projected

class TimeXLModel(nn.Module):
    # 初始化函数：定义模型所有层和参数
    def __init__(self, num_classes, k=10, input_channels=5, time_seq_len=24, time_dim=256, text_dim=64):
        # 调用父类初始化，固定写法
        super().__init__()
        # 天气分类总数（项目里是3类：雨/雪/其他）
        self.num_classes = num_classes
        
        # ===================== 1. 时间序列编码器 =====================
        # 功能：把 24小时×5个气象数值 压缩成高维特征向量
        # 输入形状：[批次B, 时间长度24, 特征数5]
        self.time_encoder = nn.Sequential(
            nn.Flatten(),          # 模型运行时，自动把数据传进来！如x = self.time_encoder(input_time_data)，# 展平：[B,24,5] → [B, 24×5=120]；
            nn.Linear(time_seq_len * input_channels, time_dim),  # 全连接层：120 → 256
            nn.ReLU(),             # 激活函数，增加非线性拟合能力
            nn.Linear(time_dim, time_dim) # 额外全连接层，不改变向量长度，只进一步提取高级特征，提升模型表达能力
        )
         
        # ===================== 2. 文本编码器 =====================
        # 功能：把天气文本描述（如小雨、多云）编码成向量
        self.text_encoder = TrainableTextEncoder(output_dim=text_dim)
        
        # ===================== 3. 特征融合层 =====================
        # 输入维度 = 时间相似度特征 + 文本相似度特征
        # C=类别数，K=每个类别的原型数量，×2是两种模态
        fusion_input_dim = num_classes * k * 2
        # 融合层：把拼接后的特征 → 映射到最终分类数
        self.fusion_layer = nn.Linear(fusion_input_dim, num_classes)

    # ===================== 核心：多模态特征融合 =====================
    def fusion(self, sim_time, sim_text):
        # 输入：
        # sim_time  [B, C*K]：时间序列与所有原型的相似度
        # sim_text  [B, C*K]：文本描述与所有原型的相似度
        combined = torch.cat([sim_time, sim_text], dim=1)  # 按维度拼接 → [B, 2*C*K]
        return self.fusion_layer(combined)  # 全连接层输出 → [B, 分类数]

class PrototypeManager(nn.Module):
    """
    原型管理器（模型最核心的可解释模块）
    功能：
        1. 维护每一类天气的「原型向量」（典型天气模式）
        2. 计算输入的时间/文本特征 与 所有原型的相似度
        3. 输出相似度，用于后续融合与预测
    核心思想：用“历史相似天气模式”来解释预测 → 可解释AI
    """
    def __init__(self, num_classes=3, k=10, time_dim=256, text_dim=64):
        # 调用父类 nn.Module 初始化
        super().__init__()
        
        # 基本参数配置
        self.num_classes = num_classes  # 天气类别数：3类（雨/雪/其他）
        self.k = k                     # 每个类别拥有多少个原型：10个/类
        self.time_dim = time_dim       # 时间特征的维度：256
        self.text_dim = text_dim       # 文本特征的维度：64

        # ===================== 核心：可学习原型参数 =====================
        # 时间模态原型：形状 [C, K, D_time] = [3, 10, 256]
        # C=类别，K=每个类的原型数量，D=特征维度
        # nn.Parameter = 把张量P_time标记为模型参数，训练时会自动更新
        # 用randn来在每一维度生成随机数，初始化原型向量
        self.P_time = nn.Parameter(torch.randn(num_classes, k, time_dim))
        
        # 文本模态原型：形状 [C, K, D_text] = [3, 10, 64]
        self.P_text = nn.Parameter(torch.randn(num_classes, k, text_dim))

    def forward(self, z_time, z_text):
        """
        前向传播：计算输入特征与所有原型的相似度
        :param z_time:  时间特征  [B, time_dim] → B=批次，256维
        :param z_text:  文本特征  [B, text_dim] → B=批次，64维
        :return: sim_time_flat, sim_text_flat → 展平后的相似度矩阵
        """
        # ===================== 计算相似度（核心公式） =====================
        # 使用 einsum 高效计算 输入特征 与 所有原型 的内积（相似度）
        # einsum 维度解释：
        # bd  = 输入 [B, D]
        # ckd = 原型 [C, K, D]
        # bck = 输出 [B, C, K] → 每个样本，对每个类、每个原型的相似度分数
        
        # 时间特征 ↔ 时间原型 相似度 → [B, 3, 10]
        sim_time = torch.einsum('bd,ckd->bck', z_time, self.P_time)
        
        # 文本特征 ↔ 文本原型 相似度 → [B, 3, 10]
        sim_text = torch.einsum('bd,ckd->bck', z_text, self.P_text)

        # ===================== 展平相似度，方便后续融合 =====================
        # 将 [B, C, K] 展平为 [B, C*K]
        # 3类 × 10原型 = 30 维特征
        
        # 时间相似度展平 [B, 3*10=30]
        sim_time_flat = sim_time.reshape(z_time.size(0), -1)
        
        # 文本相似度展平 [B, 3*10=30]
        sim_text_flat = sim_text.reshape(z_text.size(0), -1)

        # 返回：时间相似度特征、文本相似度特征（给融合层使用）
        return sim_time_flat, sim_text_flat
