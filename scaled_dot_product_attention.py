import torch
import torch.nn.functional as F
import math

def scaled_dot_production_attention(Q,K,V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(-2,-1))
    scores = scores/math.sqrt(d_k)
    atten_weights = F.softmax(scores,dim = -1)
    output = torch.matmul(atten_weights,V)

    return output , atten_weights

Q = torch.randn(2,5,4)
K = torch.randn(2,5,4)
V = torch.randn(2,5,4)

output , atten_weights = scaled_dot_production_attention(Q,K,V)

print("output为:",output)
print("atten_weights为:",atten_weights)
