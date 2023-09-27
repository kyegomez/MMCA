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
Algorithmic pseudocode

```latex
Input: Visual tokens V, Textual tokens T
Output: Updated Textual tokens T'

1: procedure MMCA(V, T)
2:     for each visual token v in V do
3:         v' = self_attention(v)  // Visual tokens only attend to themselves
4:     end for
5:     for each textual token t in T do
6:         t' = attention(t, T_previous) + attention(t, V)  // Textual tokens attend to all their previous tokens AND image tokens
7:     end for
8:     return T'
9: end procedure
```

# Multi-Modal Causal Attention: A Study

MMCA is a novel attention mechanism designed to handle multi-modal data, i.e., data that comes from different sources or formats, such as text and images. It is an extension of the causal attention mechanism, which is commonly used in transformer models for tasks like language modeling.

## Causal Attention
----------------

Before diving into MMCA, let's first understand the concept of causal attention. In the context of transformers, attention is a measure of how much a model should focus on different parts of the input when producing a particular part of the output.

Causal attention, also known as autoregressive or self-attention, is a type of attention where a token can only attend to previous tokens in the sequence. This is in contrast to other types of attention where a token can attend to all other tokens in the sequence.

The causal attention mechanism can be visualized as follows:

```
Token1 -> |------|
Token2 -> |------|------|
Token3 -> |------|------|------|
Token4 -> |------|------|------|------|

```

Each token can attend to itself and all the tokens before it, but not the ones after it.

----

## Multi-Modal Causal Attention

In a multi-modal setting, we often deal with different types of data simultaneously. For instance, in an image captioning task, the model has to process both image features and textual data. This is where MMCA comes into play.

MMCA extends the concept of causal attention to handle multi-modal data. The key idea behind MMCA is as follows:

1.  For visual tokens, they only attend to themselves, as visual tokens are encoded by the visual encoder.
2.  For textual tokens, they attend to all their previous tokens. However, they have two separate attention weight matrices for their previous textual tokens and image tokens.

This can be visualized as follows:

```
Visual Tokens:
V1 -> |------|
V2 -> |------|
V3 -> |------|

Textual Tokens:
T1 -> |------|------|------|------|
T2 -> |------|------|------|------|------|
T3 -> |------|------|------|------|------|------|

```

Here, `V1`, `V2`, and `V3` are visual tokens, and `T1`, `T2`, and `T3` are textual tokens. Each visual token only attends to itself, while each textual token attends to all previous textual and visual tokens.

----

## Mathematical Formulation

Let's now delve into the mathematical formulation of MMCA. The attention mechanism in transformers is typically computed using the dot product of query `Q` and key `K` matrices, followed by a softmax operation. In MMCA, we have two separate attention weight matrices for textual and visual tokens.

Let `Q_T` and `K_T` be the query and key matrices for textual tokens, and `Q_V` and `K_V` be the query and key matrices for visual tokens. The attention weights for textual tokens attending to previous textual tokens (`A_TT`) and visual tokens (`A_TV`) can be computed as follows:

```
A_TT = softmax(Q_T * K_T^T)
A_TV = softmax(Q_T * K_V^T)

```

The updated textual token representations can then be computed by applying these attention weights to the value `V` matrices:

```
T' = A_TT * V_T + A_TV * V_V

```

Here, `V_T` and `V_V` are the value matrices for textual and visual tokens, respectively.


## Conclusion

Multi-Modal Causal Attention is a powerful attention mechanism that extends the concept of causal attention to handle multi-modal data. It allows a model to process different types of data simultaneously and in a more efficient manner. By having separate attention weight matrices for different types of tokens, MMCA allows the model to focus on the most relevant parts of the input for each type of token, leading to improved performance on multi-modal tasks.


---

# Todo
* implement flash attention from zeta as the main attn
---

# License
MIT

---

# Citations
```bibtex
@misc{2309.14327,
Author = {Zhewei Yao and Xiaoxia Wu and Conglong Li and Minjia Zhang and Heyang Qi and Olatunji Ruwase and Ammar Ahmad Awan and Samyam Rajbhandari and Yuxiong He},
Title = {DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention},
Year = {2023},
Eprint = {arXiv:2309.14327},
}
```