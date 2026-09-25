import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder import Encoder


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes):
        super().__init__()

        # 1. 词嵌入：token ID -> 向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # 3. Encoder 堆叠
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)

        # 4. 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, mask=None):
        # input_ids: [batch, seq_len]
        X = self.embedding(input_ids)      # [batch, seq_len, d_model]
        X = self.pos_enc(X)                # [batch, seq_len, d_model]
        X, attn = self.encoder(X, mask)    # [batch, seq_len, d_model]

        # 池化：对 seq_len 取平均
        X = X.mean(dim=1)                  # [batch, d_model]

        # 分类
        logits = self.classifier(X)        # [batch, num_classes]
        return logits, attn


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    vocab_size = 1000
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_layers = 3
    num_classes = 2

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    model = TextClassifier(vocab_size, d_model, num_heads, d_ff, num_layers, num_classes)
    logits, attn = model(input_ids, mask=causal_mask)

    print("输入 input_ids shape:", input_ids.shape)  
    print("输出 logits shape:", logits.shape)         
    print("注意力 attn shape:", attn.shape)          
