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