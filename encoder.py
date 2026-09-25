import torch
import torch.nn as nn
from encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,num_heads,d_ff)
            for _ in range(num_layers)
        ])

    def forward(self,X,mask):
        for layer in self.layers:
            X,attn = layer(X,mask)

        return X,attn


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_layers = 3

    X = torch.randn(batch_size, seq_len, d_model)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    encoder = Encoder(d_model, num_heads, d_ff, num_layers)
    output, attn = encoder(X, mask=causal_mask)

    print("输入 X shape:", X.shape)
    print("输出 output shape:", output.shape)
    print("注意力 attn shape:", attn.shape)