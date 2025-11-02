import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    实现 Positional Encoding (PE) 位置编码模块

    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算 div_term，用于缩放位置
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # *** 修正点 1: 改变形状以匹配 (1, max_len, d_model) ***
        pe = pe.unsqueeze(0) # 之前是 pe.unsqueeze(0).transpose(0, 1)
        
        # register_buffer 将 pe 注册为模型的一部分
        self.register_buffer('pe', pe)

    def forward(self, x):
        # *** 修正点 2: x 的形状现在是 (batch_size, seq_len, d_model) ***
        # 截取所需长度的 PE，并将其添加到 x 上
        # self.pe[:, :x.size(1), :] 会广播到 x 的 batch_size 维度
        x = x + self.pe[:, :x.size(1), :]
        return x
    

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力
    """
    # 1. 计算 Q 和 K 的点积，获取 "scores"
    # (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码 (Mask)
    # 掩码是至关重要的。
    # - 在 Encoder/Decoder 中，用于 "padding mask"，忽略 <pad> 标记。
    # - 在 Decoder 中，用于 "look-ahead mask"，防止看到未来的标记。
    if mask is not None:
        # 将掩码中为 0 (或 False) 的位置设置为一个非常小的负数 (-1e9)
        # 这样在 softmax 之后，这些位置的概率将接近 0
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 3. 应用 Softmax 得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 4. 将权重乘以 V
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    实现 Multi-Head Attention 多头注意力模块
    """
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads # 每个头的维度
        self.n_heads = n_heads
        
        # 线性投影层
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        
        # 最后的输出线性层
        self.fc_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value 的形状: (batch_size, seq_len, d_model)
        batch_size = query.size(0)
        
        # 1. 线性投影
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # 2. 拆分为多个头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算缩放点积注意力
        # context 的形状: (batch_size, n_heads, seq_len, d_k)
        context, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 合并多头
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        # (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        context = context.view(batch_size, -1, self.n_heads * self.d_k)
        
        # 5. 最终线性投影
        output = self.fc_o(context)
        
        return output, attn_weights


class PositionWiseFFN(nn.Module):
    """
    实现 Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    """
    实现单个 Encoder Layer
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        # Add & Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x 的形状: (batch_size, seq_len, d_model)
        
        # 1. 自注意力层
        # _x 是残差连接的输入
        _x = x
        # 注意力计算
        x, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        # 残差连接 和 Dropout
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        # 2. 前馈网络层
        _x = x
        x = self.ffn(x)
        # 残差连接 和 Dropout
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        
        return x
    
class Encoder(nn.Module):
    """
    实现 N 个 EncoderLayer 堆叠的 Encoder
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # 堆叠 N 个 EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        # x 的形状: (batch_size, src_seq_len)
        
        # 1. 词嵌入 + 位置编码
        # (batch_size, src_seq_len, d_model)
        x = self.embedding(x) * math.sqrt(self.d_model) # 乘以 d_model 的平方根
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. 依次通过 N 个 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)
            
        return x # (batch_size, src_seq_len, d_model)
    

class DecoderLayer(nn.Module):
    """
    实现单个 Decoder Layer
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        # Add & Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x 的形状: (batch_size, tgt_seq_len, d_model)
        # enc_output 的形状: (batch_size, src_seq_len, d_model)
        # src_mask: 用于 Encoder 输出的 padding mask
        # tgt_mask: 用于 Decoder 输入的 look-ahead mask 和 padding mask
        
        # 1. 掩码自注意力 (Masked Self-Attention)
        _x = x
        x, _ = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        # 2. 交叉注意力 (Cross-Attention)
        _x = x
        # Q 来自 Decoder (x)，K 和 V 来自 Encoder (enc_output)
        x, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, mask=src_mask)
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        
        # 3. 前馈网络层
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(_x + x)
        
        return x

class Decoder(nn.Module):
    """
    实现 N 个 DecoderLayer 堆叠的 Decoder
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # 堆叠 N 个 DecoderLayer
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x 的形状: (batch_size, tgt_seq_len)
        # enc_output 的形状: (batch_size, src_seq_len, d_model)
        
        # 1. 词嵌入 + 位置编码
        # (batch_size, tgt_seq_len, d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. 依次通过 N 个 DecoderLayer
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x # (batch_size, tgt_seq_len, d_model)
    
class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        
        # 最终的线性层，用于生成词汇表大小的 logits
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        # 1. Encoder 编码
        # enc_output 形状: (batch_size, src_seq_len, d_model)
        enc_output = self.encoder(src, src_mask)
        
        # 2. Decoder 解码
        # dec_output 形状: (batch_size, tgt_seq_len, d_model)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # 3. 最终输出
        # output 形状: (batch_size, tgt_seq_len, tgt_vocab_size)
        output = self.final_linear(dec_output)
        
        return output