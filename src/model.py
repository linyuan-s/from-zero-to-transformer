import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) 
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads 
        self.n_heads = n_heads
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        
        self.fc_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        context, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.n_heads * self.d_k)
        
        output = self.fc_o(context)
        
        return output, attn_weights

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = x
        x, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        
        return x

class Encoder(nn.Module):
    # --- 修正 1: 在 __init__ 中添加 use_positional_encoding ---
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000, use_positional_encoding=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # --- 修正 2: 根据开关，有条件地创建 PE ---
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # --- 修正 3: 根据开关，有条件地应用 PE ---
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
            
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        _x = x
        x, _ = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        _x = x
        x, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, mask=src_mask)
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(_x + x)
        
        return x

class Decoder(nn.Module):
    # --- 修正 4: 在 __init__ 中添加 use_positional_encoding ---
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000, use_positional_encoding=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # --- 修正 5: 根据开关，有条件地创建 PE ---
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len)
            
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # --- 修正 6: 根据开关，有条件地应用 PE ---
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
            
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x

class Transformer(nn.Module):
    # --- 修正 7: __init__ 需要接收 use_positional_encoding ---
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model, 
                 n_layers, 
                 n_heads, 
                 d_ff, 
                 dropout,
                 max_len=5000,
                 use_positional_encoding=True): # <--- 添加这个参数
        super(Transformer, self).__init__()
        
        # --- 修正 8: 将 use_positional_encoding 传递下去 ---
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, use_positional_encoding)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len, use_positional_encoding)
        
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.final_linear(dec_output)
        return output