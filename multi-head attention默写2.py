import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def scaled_dot_product_attention(Q,K,V,mask):
    d_k = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1,float('-inf'))

    attn_weights = F.softmax(scores,dim = -1)

    output = torch.matmul(attn_weights,V)

    return output,attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self,X,mask):
        batch,seq_len,_ = X.size()

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K = K.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V = V.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)

        output,attn = scaled_dot_product_attention(Q,K,V,mask)

        output = output.transpose(1,2).contiguous().view(batch,seq_len,self.d_model)

        output = self.W_o(output)

        return output,attn


