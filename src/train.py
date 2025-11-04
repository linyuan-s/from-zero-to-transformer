import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import math
import argparse         
from tqdm import tqdm
from model import Transformer
from torch.optim import lr_scheduler
from datetime import datetime
from model import Transformer
from data_loader import get_data_loader
from utils import create_padding_mask, create_combined_mask

# --- 1. 定义超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 超参数建议 ---
BATCH_SIZE = 32
D_MODEL = 256       # Embedding dimension
N_HEADS = 8         # Number of heads
D_FF = 1024         # Feed-forward dimension
N_LAYERS = 4        # Number of layers
DROPOUT = 0.15
LEARNING_RATE = 3e-4
# -----------------------------

N_EPOCHS = 10       # 训练轮数 
CLIP_VALUE = 1.0    # 梯度裁剪值

# 检查点保存路径
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 结果图保存路径
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 新增: 定义命令行参数解析 ---
def parse_args():
    """
    解析命令行参数，用于消融实验
    """
    parser = argparse.ArgumentParser(description="Transformer Training with Ablation Options")
    
    # 添加一个布尔标志。如果运行时输入 --no-pe，则 args.no_pe 为 True
    parser.add_argument(
        '--no-pe', 
        action='store_true', 
        help="禁用 Positional Encoding (用于消融实验)"
    )
    return parser.parse_args()


# --- 2. 初始化所有组件 ---

def initialize_components(use_pe=True): # <--- 修改: 接收 use_pe 参数
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
        dropout=DROPOUT,
        use_positional_encoding=use_pe  # <--- 修改: 将参数传递给模型
    ).to(DEVICE)
    
    # 3. 初始化损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # 4. 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5.初始化学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    return model, train_loader, val_loader, criterion, optimizer, pad_id, scheduler

# --- 3. 训练和评估循环 (这部分无需修改) ---

def train_epoch(model, data_loader, criterion, optimizer, pad_id):
    """
    执行一个训练轮次 (epoch)
    """
    model.train() 
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch Train", leave=False)
    for src_batch, tgt_batch in progress_bar:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)
        
        tgt_input = tgt_batch[:, :-1] 
        tgt_target = tgt_batch[:, 1:] 
        
        src_mask = create_padding_mask(src_batch, pad_id)
        tgt_mask = create_combined_mask(tgt_input, pad_id)
        
        optimizer.zero_grad()
        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        
        loss = criterion(
            output.view(-1, output.shape[-1]), 
            tgt_target.reshape(-1)               
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(data_loader)

def evaluate_epoch(model, data_loader, criterion, pad_id):
    """
    执行一个评估轮次 (epoch)
    """
    model.eval() 
    epoch_loss = 0
    
    with torch.no_grad(): 
        progress_bar = tqdm(data_loader, desc=f"Epoch Eval ", leave=False)
        for src_batch, tgt_batch in progress_bar:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            
            tgt_input = tgt_batch[:, :-1]
            tgt_target = tgt_batch[:, 1:]
            
            src_mask = create_padding_mask(src_batch, pad_id)
            tgt_mask = create_combined_mask(tgt_input, pad_id)
            
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            
            loss = criterion(
                output.view(-1, output.shape[-1]),
                tgt_target.reshape(-1)
            )
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(data_loader)

# --- 4. 辅助函数：绘制损失曲线 (这部分无需修改) ---

def plot_loss_curves(train_losses, val_losses, save_path):
    """
    绘制并保存训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    
    # <--- 修改: 标题根据路径动态变化 ---
    title = 'Training and Validation Loss Curves'
    if 'NO_PE' in save_path:
        title += ' (NO Positional Encoding)'
    elif 'WITH_PE' in save_path:
        title += ' (WITH Positional Encoding)'
    plt.title(title)
    # ---------------------------------

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"\n损失曲线图已保存至: {save_path}")

# --- 5. 主训练循环 ---

def main():
    # --- 关键修改: 解析参数 ---
    args = parse_args()
    use_pe = not args.no_pe # 如果指定了 --no-pe,则 use_pe 为 False
    run_tag = "WITH_PE" if use_pe else "NO_PE" # 用于文件名
    # --------------------------

    print("开始训练...")
    print(f"设备: {DEVICE}")
    print(f"*** 实验配置: Positional Encoding = {use_pe} ***") # 打印当前配置

    #1. 为本次运行创建唯一的标识符 (时间戳)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"本次运行ID (Run ID): {timestamp}")
    
    #2. 定义本次运行的唯一保存路径
    # <--- 修改: 文件名包含 run_tag ---
    unique_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{timestamp}_{run_tag}.pt")
    unique_plot_path = os.path.join(RESULTS_DIR, f"loss_curve_{timestamp}_{run_tag}.png")
    # ---------------------------------
    
    # <--- 修改: 传递 use_pe 参数 ---
    model, train_loader, val_loader, criterion, optimizer, pad_id, scheduler = initialize_components(use_pe=use_pe)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, pad_id)
        val_loss = evaluate_epoch(model, val_loader, criterion, pad_id)
        
        # 调度器会根据验证损失自动调整学习率
        scheduler.step(val_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n--- Epoch: {epoch:02} / {N_EPOCHS} ---")
        print(f"时间: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\t训练损失: {train_loss:.3f} | 训练 PPL: {math.exp(train_loss):7.3f}")
        print(f"\t验证损失: {val_loss:.3f} | 验证 PPL: {math.exp(val_loss):7.3f}")
        
        #3. 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), unique_checkpoint_path)
            print(f"*** 验证损失降低. 模型已保存至: {unique_checkpoint_path} ***")

    #4. 训练结束后绘制图像
    plot_loss_curves(train_losses, val_losses, unique_plot_path)
    
    print("\n--- 训练完成 ---")

if __name__ == "__main__":
    main()