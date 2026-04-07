import os
import torch
import logging
import sys
from torch.utils.data import DataLoader

# -------------------------- 路径配置 --------------------------
# os.path.dirname ：获取文件所在目录路径 |  此处输入：__file__(当前脚本路径) | 输出：当前脚本的文件夹路径
# os.path.join ：拼接多个路径片段 |  此处输入：基础路径、上级目录标识 | 输出：项目根目录的相对路径
# os.path.abspath ：转换为绝对路径 |  此处输入：相对路径 | 输出：项目根目录绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# sys.path.append ：向Python导入路径添加目录 |  此处输入：项目根目录 | 输出：无，修改环境变量
# 作用：确保项目内的模块可以正常导入
sys.path.append(PROJECT_ROOT)

# 导入自定义数据集、训练器、模型、损失函数
from data.real_data_loader import ProcessedWeatherDataset
from training.base_trainer import BaseTrainer
from training.models import TimeXLModel, PrototypeManager
from training.loss import TimeXLLoss

# -------------------------- 日志配置 --------------------------
# logging.basicConfig ：配置日志的基础参数 |  此处输入：日志级别、格式、处理器 | 输出：无，全局配置日志系统
# handlers配置：同时输出日志到控制台和文件
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),       # 控制台输出日志
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'training.log'))  # 文件保存日志
    ]
)
# logging.getLogger ：创建/获取日志记录器 |  此处输入：日志器名称 | 输出：日志器对象
logger = logging.getLogger("EncoderTrainer")

# -------------------------- 核心训练函数 --------------------------
def train_encoder():
    # logger.info ：输出INFO级别日志 |  此处输入：日志文本 | 输出：无，打印日志
    logger.info("=== Starting Prototype-based Encoder Training ===")
    
    # -------------------------- 超参数配置 --------------------------
    # os.path.join ：拼接路径 |  此处输入：项目根目录、data文件夹名 | 输出：数据文件夹完整路径
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CITY = 'San Francisco'       # 训练的目标城市
    BATCH_SIZE = 64              # 批次大小
    EPOCHS = 10                  # 训练轮数
    LR = 0.001                   # 学习率
    # torch.cuda.is_available ：判断GPU是否可用 |  此处输入：无 | 输出：布尔值
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Device: {DEVICE}")
    
    # -------------------------- 1. 加载数据集 --------------------------
    logger.info(f"Loading data for {CITY}...")
    try:
        # ProcessedWeatherDataset ：加载预处理后的气象数据集 |  此处输入：数据路径、城市、数据集划分 | 输出：数据集对象
        train_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='train')
        val_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='val')
        test_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='test')
    except FileNotFoundError as e:
        # logger.error ：输出ERROR级别日志 |  此处输入：错误信息 | 输出：无，打印错误日志
        logger.error(f"Data not found: {e}")
        logger.error("Please run data/preprocess_data.py first.")
        return

    # DataLoader ：封装数据集为迭代器 |  此处输入：数据集、批次大小、是否打乱 | 输出：数据加载器对象
    # 训练集打乱，验证/测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # len() ：获取容器长度 |  此处输入：标签映射字典 | 输出：天气类别总数
    num_classes = len(train_dataset.label_map)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples: {len(train_dataset)}")
    
    # -------------------------- 2. 初始化模型组件 --------------------------
    k = 10 # 每个类别对应的原型数量
    # TimeXLModel ：初始化多模态编码模型 |  此处输入：类别数、每类原型数 | 输出：模型对象
    model = TimeXLModel(num_classes=num_classes, k=k)
    # PrototypeManager ：初始化原型管理器 |  此处输入：类别数、每类原型数 | 输出：原型管理器对象
    pm = PrototypeManager(num_classes=num_classes, k=k)
    # TimeXLLoss ：初始化自定义损失函数 |  此处输入：无 | 输出：损失函数对象
    loss_fn = TimeXLLoss()
    
    # model.to() ：将模型移动到指定设备 |  此处输入：cpu/cuda | 输出：无，修改模型存储位置
    model.to(DEVICE)
    pm.to(DEVICE)
    
    # torch.optim.Adam ：初始化Adam优化器 |  此处输入：模型参数、学习率 | 输出：优化器对象
    # list() + 合并：将两个模型的参数合并，统一优化
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pm.parameters()), 
        lr=LR
    )
    
    # BaseTrainer ：初始化训练器 |  此处输入：模型、原型管理器、损失函数、优化器、设备 | 输出：训练器对象
    trainer = BaseTrainer(model, pm, loss_fn, optimizer, device=DEVICE)
    
    # -------------------------- 3. 训练循环 --------------------------
    best_val_acc = 0.0  # 记录最优验证集准确率
    
    logger.info("Starting Training Loop...")
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # trainer.train_epoch ：执行一轮训练 |  此处输入：训练数据加载器 | 输出：训练损失、准确率、其他指标
        train_loss, train_acc, _ = trainer.train_epoch(train_loader)
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # trainer.validate ：执行验证/测试 |  此处输入：验证数据加载器 | 输出：验证损失、准确率
        val_loss, val_acc = trainer.validate(val_loader)
        logger.info(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
        
        # 保存最优模型：当前准确率大于历史最优时保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New Best Accuracy! Saving model...")
            # torch.save ：保存模型权重 |  此处输入：模型参数字典、保存路径 | 输出：无，生成权重文件
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, 'pth', 'best_encoder.pth'))
            torch.save(pm.state_dict(), os.path.join(PROJECT_ROOT, 'pth', 'best_prototypes.pth'))
            
    # -------------------------- 4. 最终测试 --------------------------
    logger.info("\n=== Final Testing ===")
    # 拼接最优模型路径
    best_encoder_path = os.path.join(PROJECT_ROOT, 'best_encoder.pth')
    best_prototypes_path = os.path.join(PROJECT_ROOT, 'best_prototypes.pth')
    
    # os.path.exists ：判断文件是否存在 |  此处输入：文件路径 | 输出：布尔值
    if os.path.exists(best_encoder_path):
        # torch.load ：加载模型权重 |  此处输入：权重文件路径 | 输出：参数字典
        # model.load_state_dict ：加载参数到模型 |  此处输入：参数字典 | 输出：无
        model.load_state_dict(torch.load(best_encoder_path))
        pm.load_state_dict(torch.load(best_prototypes_path))
        
    # 在测试集上评估最优模型
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
# -------------------------- 主函数入口 --------------------------
if __name__ == "__main__":
    # 调用训练函数，启动训练
    train_encoder()