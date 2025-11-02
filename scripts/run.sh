#!/bin/bash

# 设置严格模式
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

echo "\n[步骤 1/3] 检查/安装 依赖..."
# 确保所有依赖都已安装
pip install -r requirements.txt

echo "\n[步骤 2/3] 训练分词器 (Tokenizer)..."
# 运行 build_tokenizer.py 来创建 tokenizers/
python src/build_tokenizer.py

echo "\n[步骤 3/3] 开始训练..."
# 运行主训练脚本
python src/train.py

echo "\n--- 运行完成 ---"
echo "训练结果图已保存至: results/"
echo "最佳模型已保存至: checkpoints/"
echo "============================================="