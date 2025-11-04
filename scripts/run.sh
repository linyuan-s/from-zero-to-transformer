#!/bin/bash

# 设置严格模式，任何命令失败都会使脚本停止
set -e

echo "============================================="
echo "大模型作业：从零实现 Transformer (Encoder-Decoder)"
echo "============================================="

# 检查 Python 环境
if ! command -v python &> /dev/null
then
    echo "错误: 未找到 'python' 命令。"
    echo "请确保你已激活正确的 conda 环境 (例如: conda activate transformer)"
    exit 1
fi

echo "\n[步骤 1/4] 检查/安装 依赖..."
# 确保所有依赖都已安装
pip install -r requirements.txt

echo "\n[步骤 2/4] 训练分词器 (Tokenizer)..."
# 运行 build_tokenizer.py 来创建 tokenizers/
# (这个脚本 已经存在，我们直接调用)
python src/build_tokenizer.py

# --- 消融实验 ---

echo "\n[步骤 3/4] 开始训练: 优化模型 (带位置编码 + 调度器)"
echo "这是我们的最佳模型..."
# (我们运行 `src/train.py`，不带任何参数，
#  默认 use_pe=True)
python src/train.py

echo "\n[步骤 4/4] 开始训练: 消融实验 (无位置编码)"
echo "这将复现报告中的图 'training_validation_loss.png'..."
# (我们运行同一个脚本，但传入 --no-pe 参数)
python src/train.py --no-pe

echo "\n--- 所有训练运行完成 ---"
echo "训练结果图已保存至: results/"
echo "最佳模型已保存至: checkpoints/"
echo "============================================="