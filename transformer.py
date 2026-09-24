import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ====================== 1. 缩放点积注意力实现 ======================
class ScaledDotProductAttention(nn.Module):
    """
    论文3.2.1节：缩放点积注意力
    公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        :param q: 查询张量，shape: [batch_size, n_heads, seq_len, d_k]
        :param k: 键张量，shape: [batch_size, n_heads, seq_len, d_k]
        :param v: 值张量，shape: [batch_size, n_heads, seq_len, d_v]
        :param mask: 掩码张量，shape: [batch_size, 1, seq_len, seq_len]，需要mask的位置为True
        :return: 注意力输出，注意力权重矩阵
        """
        d_k = q.size(-1)
        
        # 1. 计算Q和K的点积，除以sqrt(d_k)缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用掩码：需要mask的位置，设置为-1e9，softmax后权重趋近于0
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        # 3. 计算softmax，得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. 应用dropout
        attn_weights = self.dropout(attn_weights)
        
        # 5. 注意力权重和V相乘，得到最终输出
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

# ====================== 2. 多头注意力实现 ======================
class MultiHeadAttention(nn.Module):
    """
    论文3.2.2节：多头注意力机制
    公式：MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    """
    def __init__(self, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        # 论文参数：d_model=512，n_heads=8，每个头的维度d_k = d_model / n_heads = 64
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层：Q、K、V的投影，以及最终的输出投影
        self.w_q = nn.Linear(d_model, d_model)  # W_Q: [d_model, d_model]
        self.w_k = nn.Linear(d_model, d_model)  # W_K: [d_model, d_model]
        self.w_v = nn.Linear(d_model, d_model)  # W_V: [d_model, d_model]
        self.w_o = nn.Linear(d_model, d_model)  # W_O: [d_model, d_model]
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
        
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        :param q: 查询张量，shape: [batch_size, seq_len_q, d_model]
        :param k: 键张量，shape: [batch_size, seq_len_k, d_model]
        :param v: 值张量，shape: [batch_size, seq_len_v, d_model]
        :param mask: 掩码张量，shape: [batch_size, seq_len_q, seq_len_k]
        :return: 多头注意力输出，注意力权重矩阵
        """
        batch_size = q.size(0)
        
        # 1. 线性投影，把Q、K、V投影到d_model维度
        q = self.w_q(q)  # [batch_size, seq_len_q, d_model]
        k = self.w_k(k)  # [batch_size, seq_len_k, d_model]
        v = self.w_v(v)  # [batch_size, seq_len_v, d_model]
        
        # 2. 把张量拆分成n_heads个头，shape变成: [batch_size, n_heads, seq_len, d_k]
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 3. 缩放点积注意力计算
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # 4. 把多个头的结果拼接回来，shape变回: [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终的线性投影
        output = self.w_o(attn_output)
        
        # 6. 应用dropout
        output = self.dropout(output)
        
        return output, attn_weights

# ====================== 3. 逐位置前馈网络实现 ======================
class PositionWiseFeedForward(nn.Module):
    """
    论文3.3节：逐位置前馈网络
    公式：FFN(x) = max(0, xW1 + b1) W2 + b2
    """
    def __init__(self, 
                 d_model: int = 512, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        # 两个线性层，中间ReLU激活
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对每个位置独立执行前馈计算
        :param x: 输入张量，shape: [batch_size, seq_len, d_model]
        :return: 输出张量，shape和输入一致
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ====================== 4. 位置编码实现 ======================
class PositionalEncoding(nn.Module):
    """
    论文3.5节：正弦余弦位置编码
    """
    def __init__(self, 
                 d_model: int = 512, 
                 max_seq_len: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 预计算所有位置的位置编码
        pe = torch.zeros(max_seq_len, d_model)
        # 位置索引，shape: [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用正弦函数，奇数维度用余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度，shape: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 把pe注册为buffer，不会被优化器更新，但会被保存到模型文件里
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：把位置编码加到输入嵌入向量上
        :param x: 输入嵌入向量，shape: [batch_size, seq_len, d_model]
        :return: 加入位置编码后的向量
        """
        seq_len = x.size(1)
        # 取对应长度的位置编码，加到输入上
        x = x + self.pe[:, :seq_len, :].detach()
        # 应用dropout
        x = self.dropout(x)
        return x

# ====================== 5. 编码器层实现 ======================
class EncoderLayer(nn.Module):
    """
    论文3.1节：单个编码器层
    包含两个子层：多头自注意力层 + 逐位置前馈网络
    每个子层都有残差连接 + 层归一化
    """
    def __init__(self, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        # 多头自注意力层
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 逐位置前馈网络
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param x: 输入张量，shape: [batch_size, seq_len, d_model]
        :param mask: 自注意力掩码，padding mask
        :return: 编码器层输出
        """
        # 子层1：多头自注意力，Pre-LN结构
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, mask)
        x = residual + x  # 残差连接
        
        # 子层2：前馈网络，Pre-LN结构
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x  # 残差连接
        
        return x

# ====================== 6. 解码器层实现 ======================
class DecoderLayer(nn.Module):
    """
    论文3.1节：单个解码器层
    包含三个子层：掩码多头自注意力层 + 编码器-解码器注意力层 + 逐位置前馈网络
    每个子层都有残差连接 + 层归一化
    """
    def __init__(self, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        # 1. 带掩码的多头自注意力层
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 2. 编码器-解码器注意力层（交叉注意力）
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 3. 逐位置前馈网络
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                enc_output: torch.Tensor, 
                look_ahead_mask: torch.Tensor = None, 
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param x: 解码器输入，shape: [batch_size, tgt_seq_len, d_model]
        :param enc_output: 编码器的输出，shape: [batch_size, src_seq_len, d_model]
        :param look_ahead_mask: 前瞻掩码，防止看到未来的token
        :param padding_mask: padding掩码，用于交叉注意力，屏蔽编码器的padding token
        :return: 解码器层输出
        """
        # 子层1：掩码多头自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, look_ahead_mask)
        x = residual + x
        
        # 子层2：编码器-解码器交叉注意力
        residual = x
        x = self.norm2(x)
        # Q来自解码器，K和V来自编码器输出
        x, _ = self.cross_attn(x, enc_output, enc_output, padding_mask)
        x = residual + x
        
        # 子层3：前馈网络
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

# ====================== 7. 完整编码器实现 ======================
class Encoder(nn.Module):
    """
    完整的编码器：N个编码器层堆叠
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 d_model: int = 512, 
                 n_layers: int = 6, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 max_seq_len: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # token嵌入层
        self.token_embedding = nn.Embedding(src_vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        # N个编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param x: 输入token序列，shape: [batch_size, src_seq_len]
        :param mask: 掩码张量
        :return: 编码器最终输出
        """
        # 1. token嵌入，乘以sqrt(d_model)缩放，和论文一致
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        # 2. 加入位置编码
        x = self.pos_encoding(x)
        # 3. 依次通过N个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        # 4. 最终的层归一化
        x = self.norm(x)
        return x

# ====================== 8. 完整解码器实现 ======================
class Decoder(nn.Module):
    """
    完整的解码器：N个解码器层堆叠
    """
    def __init__(self, 
                 tgt_vocab_size: int, 
                 d_model: int = 512, 
                 n_layers: int = 6, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 max_seq_len: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # token嵌入层
        self.token_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        # N个解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, 
                x: torch.Tensor, 
                enc_output: torch.Tensor, 
                look_ahead_mask: torch.Tensor = None, 
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param x: 目标token序列，shape: [batch_size, tgt_seq_len]
        :param enc_output: 编码器输出
        :param look_ahead_mask: 前瞻掩码
        :param padding_mask: padding掩码
        :return: 解码器最终输出
        """
        # 1. token嵌入，乘以sqrt(d_model)缩放
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        # 2. 加入位置编码
        x = self.pos_encoding(x)
        # 3. 依次通过N个解码器层
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
        # 4. 最终的层归一化
        x = self.norm(x)
        return x

# ====================== 9. 完整Transformer模型实现 ======================
class Transformer(nn.Module):
    """
    完整的Transformer模型，和论文参数完全对齐
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int = 512, 
                 n_layers: int = 6, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 max_seq_len: int = 5000, 
                 dropout: float = 0.1,
                 pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        # 编码器
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout)
        # 解码器
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout)
        # 最终的线性层+softmax，预测下一个token
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # 论文里的权重共享：嵌入层和最终线性层共享权重
        self.encoder.token_embedding.weight = self.fc.weight
        self.decoder.token_embedding.weight = self.fc.weight
        
        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """
        按照论文的方式，初始化模型参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        创建padding掩码：padding的位置为True，需要被mask
        :param seq: 输入序列，shape: [batch_size, seq_len]
        :return: 掩码张量，shape: [batch_size, 1, 1, seq_len]
        """
        mask = (seq == self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask

    def create_look_ahead_mask(self, seq_len: int) -> torch.Tensor:
        """
        创建前瞻掩码：上三角矩阵，未来的位置为True，需要被mask
        :param seq_len: 序列长度
        :return: 掩码张量，shape: [1, 1, seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param src_seq: 源序列，shape: [batch_size, src_seq_len]
        :param tgt_seq: 目标序列，shape: [batch_size, tgt_seq_len]
        :return: 模型输出，shape: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 1. 创建编码器的padding掩码
        enc_padding_mask = self.create_padding_mask(src_seq)
        # 2. 编码器前向传播
        enc_output = self.encoder(src_seq, enc_padding_mask)
        
        # 3. 创建解码器的掩码
        # 3.1 前瞻掩码，防止看到未来的token
        tgt_seq_len = tgt_seq.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt_seq.device)
        # 3.2 目标序列的padding掩码
        tgt_padding_mask = self.create_padding_mask(tgt_seq)
        # 3.3 合并两个掩码：只要有一个需要mask，就mask
        dec_self_attn_mask = torch.max(look_ahead_mask, tgt_padding_mask)
        
        # 4. 交叉注意力的padding掩码，屏蔽编码器的padding token
        dec_cross_attn_mask = self.create_padding_mask(src_seq)
        
        # 5. 解码器前向传播
        dec_output = self.decoder(tgt_seq, enc_output, dec_self_attn_mask, dec_cross_attn_mask)
        
        # 6. 最终的线性层，预测每个位置的token概率
        output = self.fc(dec_output)
        
        return output

# ====================== 模型测试 ======================
if __name__ == "__main__":
    # 超参数，和论文完全一致
    src_vocab_size = 37000  # 源语言词汇表大小，和论文BPE分词一致
    tgt_vocab_size = 37000  # 目标语言词汇表大小
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    max_seq_len = 512
    dropout = 0.1
    pad_token_id = 0

    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_token_id=pad_token_id
    )

    # 生成测试数据：batch_size=2，源序列长度10，目标序列长度8
    batch_size = 2
    src_seq = torch.randint(1, src_vocab_size, (batch_size, 10))  # 1是有效token，0是padding
    tgt_seq = torch.randint(1, tgt_vocab_size, (batch_size, 8))

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(src_seq, tgt_seq)

    # 打印输出形状
    print(f"模型输入源序列形状: {src_seq.shape}")
    print(f"模型输入目标序列形状: {tgt_seq.shape}")
    print(f"模型输出形状: {output.shape}")
    print("✅ 模型前向传播测试通过！")
