import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import math
from tqdm import tqdm
from model import Transformer

# --- 从我们自己的 .py 文件中导入 ---
from model import Transformer
from data_loader import get_data_loader
from utils import create_padding_mask, create_combined_mask

# --- 1. 定义超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 超参数建议 ---
BATCH_SIZE = 32
D_MODEL = 256       # Embedding dimension
N_HEADS = 8         # Number of heads
D_FF = 512          # Feed-forward dimension
N_LAYERS = 4        # Number of layers
DROPOUT = 0.1
LEARNING_RATE = 3e-4
# -----------------------------

N_EPOCHS = 10       # 训练轮数 (可以先设为10，跑通后再增加)
CLIP_VALUE = 1.0    # 梯度裁剪值

# 检查点保存路径
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# 结果图保存路径
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_PATH = os.path.join(RESULTS_DIR, "training_validation_loss.png")


# --- 2. 初始化所有组件 ---

def initialize_components():
    """
    加载数据, 初始化模型, 损失函数和优化器
    """
    # 1. 加载数据
    print("加载数据...")
    train_dataset, train_loader = get_data_loader('train', BATCH_SIZE)
    val_dataset, val_loader = get_data_loader('validation', BATCH_SIZE)
    
    # 获取词表大小和 pad_id
    src_vocab_size = train_dataset.src_tokenizer.get_vocab_size()
    tgt_vocab_size = train_dataset.tgt_tokenizer.get_vocab_size()
    pad_id = train_dataset.tgt_pad_id
    
    print(f"源 (en) 词表大小: {src_vocab_size}")
    print(f"目标 (de) 词表大小: {tgt_vocab_size}")
    print(f"Padding ID: {pad_id}")

    # 2. 初始化模型
    print("初始化模型...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # 3. 初始化损失函数
    # CrossEntropyLoss 自动处理 logits 和 softmax
    # ignore_index=pad_id 会让损失函数忽略所有 <pad> 标记
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # 4. 初始化优化器
    # AdamW 是 Adam 的一个改进版本，有助于防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    return model, train_loader, val_loader, criterion, optimizer, pad_id

# --- 3. 训练和评估循环 ---

def train_epoch(model, data_loader, criterion, optimizer, pad_id):
    """
    执行一个训练轮次 (epoch)
    """
    model.train() # 设置为训练模式
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch Train", leave=False)
    for src_batch, tgt_batch in progress_bar:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)
        
        # 1. 准备模型输入和目标
        tgt_input = tgt_batch[:, :-1] # (batch, tgt_len - 1)
        tgt_target = tgt_batch[:, 1:] # (batch, tgt_len - 1)
        
        # 2. 创建掩码
        src_mask = create_padding_mask(src_batch, pad_id)
        tgt_mask = create_combined_mask(tgt_input, pad_id)
        
        # 3. 前向传播
        optimizer.zero_grad()
        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        # output 形状: (batch, tgt_len - 1, tgt_vocab_size)
        
        # 4. 计算损失
        # CrossEntropyLoss 期望 (N, C, ...) 和 (N, ...)
        # N = batch_size * (tgt_len - 1)
        # C = tgt_vocab_size
        loss = criterion(
            output.view(-1, output.shape[-1]), # (batch * (tgt_len-1), tgt_vocab_size)
            tgt_target.reshape(-1)               # (batch * (tgt_len-1))
        )
        
        # 5. 反向传播和优化
        loss.backward()

        progress_bar.set_postfix(loss=loss.item())

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(data_loader)

def evaluate_epoch(model, data_loader, criterion, pad_id):
    """
    执行一个评估轮次 (epoch)
    """
    model.eval() # 设置为评估模式
    epoch_loss = 0
    
    with torch.no_grad(): # 在评估期间不计算梯度
        progress_bar = tqdm(data_loader, desc=f"Epoch Eval ", leave=False)
        for src_batch, tgt_batch in progress_bar:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            
            # 1. 准备输入和目标
            tgt_input = tgt_batch[:, :-1]
            tgt_target = tgt_batch[:, 1:]
            
            # 2. 创建掩码
            src_mask = create_padding_mask(src_batch, pad_id)
            tgt_mask = create_combined_mask(tgt_input, pad_id)
            
            # 3. 前向传播
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            
            # 4. 计算损失
            loss = criterion(
                output.view(-1, output.shape[-1]),
                tgt_target.reshape(-1)
            )
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(data_loader)

# --- 4. 辅助函数：绘制损失曲线 ---

def plot_loss_curves(train_losses, val_losses):
    """
    绘制并保存训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"\n损失曲线图已保存至: {PLOT_PATH}")

# --- 5. 主训练循环 ---

def main():
    print("开始训练...")
    print(f"设备: {DEVICE}")
    
    model, train_loader, val_loader, criterion, optimizer, pad_id = initialize_components()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, pad_id)
        val_loss = evaluate_epoch(model, val_loader, criterion, pad_id)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n--- Epoch: {epoch:02} / {N_EPOCHS} ---")
        print(f"时间: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\t训练损失: {train_loss:.3f} | 训练 PPL: {math.exp(train_loss):7.3f}")
        print(f"\t验证损失: {val_loss:.3f} | 验证 PPL: {math.exp(val_loss):7.3f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"*** 验证损失降低. 模型已保存至: {CHECKPOINT_PATH} ***")

    # 训练结束后绘制图像
    plot_loss_curves(train_losses, val_losses)
    
    print("\n--- 训练完成 ---")

if __name__ == "__main__":
    main()