import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,X,mask):
        attn_out,attn_weights = self.attn(X,mask)

        X = self.norm1(X + attn_out)

        ffn_out = self.ffn(X)

        X = self.norm2(X + ffn_out)

        return X,attn_weights

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2
    d_ff = 32

    X = torch.randn(batch_size, seq_len, d_model)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    layer = EncoderLayer(d_model, num_heads, d_ff)
    output, attn = layer(X, mask=causal_mask)

    print("输入 X shape:", X.shape)
    print("输出 output shape:", output.shape)
    print("注意力 attn shape:", attn.shape)
