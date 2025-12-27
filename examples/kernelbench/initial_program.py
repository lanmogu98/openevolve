"""
KernelBench Level 3 Problem 43: MinGPT Causal Attention
========================================================

Initial program for OpenEvolve optimization.
Goal: Create an optimized CUDA kernel that matches the reference Model output
while achieving better performance.

Reference Model (from KernelBench):
- Multi-head masked self-attention
- Input: (batch_size, seq_len, n_embd) 
- Output: (batch_size, seq_len, n_embd)
- Operations: Q/K/V projection -> attention -> output projection

Baseline Performance:
- PyTorch Eager: ~35 ms
- torch.compile: ~29 ms

Your task: Optimize the ModelNew class to run faster while maintaining correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# EVOLVE-BLOCK-START

class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention.
    
    This is the class that will be evolved by OpenEvolve.
    It must:
    1. Accept the same __init__ parameters as the reference Model
    2. Produce the same output shape and values (within tolerance)
    3. Run faster than the reference implementation
    
    Current implementation: Baseline (same as reference)
    TODO: Optimize using techniques like:
    - Flash Attention (F.scaled_dot_product_attention)
    - Custom CUDA kernels (torch.utils.cpp_extension.load_inline)
    - Memory-efficient attention patterns
    - Fused operations
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# EVOLVE-BLOCK-END

