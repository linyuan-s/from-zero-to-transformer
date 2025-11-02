import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_padding_mask(seq, pad_token_id):
    """
    创建 padding mask
    这个掩码用于在自注意力和交叉注意力中忽略 <pad> 标记。
    
    输入:
    - seq: 形状为 (batch_size, seq_len) 的张量
    - pad_token_id: <pad> 标记的 ID
    
    输出:
    - mask: 形状为 (batch_size, 1, 1, seq_len) 的布尔张量。
            在 <pad> 标记的位置为 False (0)，其他位置为 True (1)。
            (多出来的维度是为了匹配多头注意力的 (batch, n_heads, seq_len, seq_len) 形状)
    """
    # 找出所有不等于 pad_token_id 的位置 (这些是 True)
    mask = (seq != pad_token_id)
    # 增加维度以匹配注意力头的广播需求
    # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
    return mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(seq_len):
    """
    创建 look-ahead mask (前瞻掩码)
    这个掩码用于 Decoder 的自注意力，防止它“偷看”未来的标记。
    
    输入:
    - seq_len: 目标序列的长度
    
    输出:
    - mask: 形状为 (1, 1, seq_len, seq_len) 的布尔张量。
            这是一个上三角矩阵，对角线及以下为 True (1)，以上为 False (0)。
    """
    # torch.tril 创建一个下三角矩阵，填充为 1
    # (seq_len, seq_len)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    # (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)


def create_combined_mask(tgt_seq, pad_token_id):
    """
    为 Decoder 创建组合掩码
    
    这结合了 padding_mask 和 look_ahead_mask。
    - padding_mask 确保不关注 <pad> 标记。
    - look_ahead_mask 确保不关注未来的标记。
    
    输入:
    - tgt_seq: 目标序列张量, 形状 (batch_size, seq_len)
    - pad_token_id: <pad> 标记的 ID
    
    输出:
    - mask: 形状为 (batch_size, 1, seq_len, seq_len) 的组合掩码。
    """
    
    # 1. 创建 Padding Mask (形状: batch_size, 1, 1, seq_len)
    # 这是为了在交叉注意力中让 Encoder 忽略 pad
    # 也为了在自注意力中让 Decoder 忽略 pad
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)
    
    # 2. 创建 Look-Ahead Mask (形状: 1, 1, seq_len, seq_len)
    seq_len = tgt_seq.shape[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)
    
    # 3. 组合掩码
    # 使用 & (逻辑与) 运算
    # padding_mask 会被广播到 (batch_size, 1, seq_len, seq_len)
    # look_ahead_mask 会被广播到 (batch_size, 1, seq_len, seq_len)
    # 最终结果是 (batch_size, 1, seq_len, seq_len)
    # 只有当一个位置既不是 padding 也不是 future时，才为 True
    return padding_mask & look_ahead_mask