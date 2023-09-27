[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Multi-Modal Causal Attention
The open source community's implementation of the all-new Multi-Modal Causal Attention from "DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention"


[Paper Link](https://arxiv.org/pdf/2309.14327.pdf)

# Appreciation
* Lucidrains
* Agorians



# Install
`pip install mmca`

# Usage
```python
import torch 
from mmca.main import MultiModalCausalAttention


attn = MultiModalCausalAttention(dim=512, heads=8)

x = torch.randn(1, 10, 512)
y = torch.randn(1, 20, 512)

x, y = attn(x, y)

print(x)
print(y)
```

# Architecture

# Todo


# License
MIT

# Citations

