import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def scaled_dot_product_attention(Q,K,V,mask):
    d_k = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(-2,-1))
    scores = scores/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1,float('-inf'))
    attn_weights = F.softmax(scores,dim = -1)
    output = torch.matmul(attn_weights,V)

    return output , attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads ==0 #检查能否整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义四个线性层
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self,X,mask = None):

        batch,seq_len,_ = X.size()

        # 根据输入的X计算Q K V
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # 拆多头 并将1 2维数据换位
        Q = Q.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K = K.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V = V.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)

        # 分别计算各头注意力
        output,attn = scaled_dot_product_attention(Q,K,V,mask)

        # 合并各多头
        output = output.transpose(1,2).contiguous().view(batch,seq_len,self.d_model)

        # 将结果进行线性层加工 真正融合信息
        output = self.W_o(output)

        return output,attn


# --------------------测试----------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2

    X = torch.randn(batch_size, seq_len, d_model)   # [2, 5, 8]

    # causal mask：上三角为 1，屏蔽未来
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # [5, 5]

    mha = MultiHeadAttention(d_model, num_heads)
    output, attn = mha(X, mask=causal_mask)

    print("输入 X shape:", X.shape)            # [2, 5, 8]
    print("输出 output shape:", output.shape)  # [2, 5, 8]
    print("注意力 attn shape:", attn.shape)    # [2, 2, 5, 5]

    print("\n第一个样本、第一个头的注意力矩阵:")
    print(attn[0, 0])

    print("\n每行和:", attn[0, 0].sum(dim=-1))


