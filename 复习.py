import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q,k,v,mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1,float('-inf'))

    # 注意缩进
    attn_weights = F.softmax(scores,dim = -1)
    output = torch.matmul(attn_weights,v)

    return output,attn_weights
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

    def forward(self,X,mask=None):
        batch,seq_len,_ = X.size()

        q = self.w_q(X)
        k = self.w_k(X)
        v = self.w_v(X)

        q = q.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        k = k.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        v = v.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)

        output,attn = scaled_dot_product_attention(q,k,v,mask)

        output = output.transpose(1,2).contiguous().view(batch,seq_len,self.d_model)

        output = self.w_o(output)

        return output,attn