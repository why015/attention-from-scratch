import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len = 5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len,d_model)

    def forward(self,X):
        seq_len = X.size(1)
        positions = torch.arange(seq_len,device = X.device).unsqueeze(0)
        return X + self.embedding(positions)

if __name__ == "__main__":
    torch.manual_seed(0)

    pos_enc = PositionalEncoding(d_model=8)
    X = torch.randn(2, 5, 8)

    output = pos_enc(X)

    print("输入 shape:", X.shape)          # [2, 5, 8]
    print("输出 shape:", output.shape)      # [2, 5, 8]