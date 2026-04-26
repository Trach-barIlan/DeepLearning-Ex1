from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    return nn.Linear(input_vector_dim, 3 * input_vector_dim)

def kqv(x, linear):
    B, N, D = x.size()
    # TODO compute k, q, and v
    # (can do it in 1 or 2 lines.)
    kqv_proj = linear(x)
    k, q, v = torch.chunk(kqv_proj, 3, dim=-1)
    return k, q, v

def attention_scores(a, b):

    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    A = (b @ a.transpose(-2, -1)) / math.sqrt(D1)
    return A

# we know that the embed_dim and n_heads are redundant parameters but we still kept them
def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.

    mask = torch.tril(torch.ones(max_context_len, max_context_len)).unsqueeze(0)
    return mask

def self_attention(v, A, mask = None):
    # Optional causal masking: locations with 0 in the mask are blocked.
    if mask is not None:
        A = A.masked_fill(mask[..., :A.size(-2), :A.size(-1)] == 0, float("-inf"))

    weights = F.softmax(A, dim=-1)
    sa = weights @ v
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    n_heads = len(kqv_matrices)
    head_dim = D // n_heads
    
    heads_output = []
    for i, kqv_matrix in enumerate(kqv_matrices):
        # Extract this head's portion of the embedding
        x_head = x[:, :, i*head_dim:(i+1)*head_dim]
        # Apply self-attention for this head
        sa_head = self_attention_layer(x_head, kqv_matrix, mask)
        heads_output.append(sa_head)
    
    # Concatenate all heads
    sa = torch.cat(heads_output, dim=-1)
    
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        head_dim = embed_dim // n_heads
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(head_dim, n_heads) for i in range(n_heads)])
        # Final linear projection after concatenating heads
        self.final_projection = nn.Linear(embed_dim, embed_dim)
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.final_projection(sa)
        return sa
