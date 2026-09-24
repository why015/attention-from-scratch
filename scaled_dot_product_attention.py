import torch
import torch.nn.functional as F
import math

def scaled_dot_production_attention(Q,K,V,mask):
    d_k = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(-2,-1))
    scores = scores/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1,float('-inf'))
    atten_weights = F.softmax(scores,dim = -1)
    output = torch.matmul(atten_weights,V)

    return output , atten_weights

seq_len = 5
casual_mask = torch.triu(torch.ones(seq_len,seq_len),diagonal = 1)

Q = torch.randn(2,5,4)
K = torch.randn(2,5,4)
V = torch.randn(2,5,4)

output , atten_weights = scaled_dot_production_attention(Q,K,V,mask = casual_mask)

print("output为:",output)
print("atten_weights为:",atten_weights)
