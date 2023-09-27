import torch 
from mmca.main import SimpleMMCA

# Define the dimensions
dim = 512
head = 8
seq_len = 10
batch_size = 32

#attn
attn = SimpleMMCA(dim=dim, heads=head)

#random tokens
v = torch.randn(batch_size, seq_len, dim)
t = torch.randn(batch_size, seq_len, dim)

#pass the tokens throught attn
tokens = attn(v, t)

print(tokens)