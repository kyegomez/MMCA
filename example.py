import torch
from mmca.main import MultiModalCausalAttention


attn = MultiModalCausalAttention(dim=512, heads=8)

x = torch.randn(1, 10, 512)
y = torch.randn(1, 20, 512)

# create a mask for the text
# mask = torch.ones(1, 20).bool()

x, y = attn(x, y)

print(x)
# print(y)
