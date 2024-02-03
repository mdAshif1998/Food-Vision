import torch
from torch import nn
from torch.nn import functional as f
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_projection_bias: bool = True, out_projection_bias: bool = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_projection_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_projection_bias)

        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask: bool = False):
        # x: (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # Mask where the upper triangle (above the principle diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -infinity
            weight.massked_fill(mask, -torch.inf)

        # Divide by d_k (Dim / H).
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_heads)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = f.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(interim_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_head: int, d_embd: int, d_cross: int, in_projection_bias: bool = True, out_projection_bias: bool = True):
        super().__init__()
        self.q_projection = nn.Linear(d_embd, d_embd, bias=in_projection_bias)
        self.k_projection = nn.Linear(d_embd, d_embd, bias=in_projection_bias)
        self.v_projection = nn.Linear(d_embd, d_embd, bias=in_projection_bias)

        self.out_projection = nn.Linear(d_embd, d_embd, bias=out_projection_bias)
        self.n_head = n_head
        self.d_head = d_embd // n_head

    def forward(self, x, y):
        # x: Latent: (Batch_Size, Seq_Len_Q, Dim_Q)
        # y: context: (Batch_Size, Seq_Len_KV, DIM_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embd = input_shape
        interim_shape = (batch_size, -1, self.n_head, self.n_head)

        # Multiply Query by Wq
        q = self.q_projection(x)
        k = self.k_projection(y)
        v = self.v_projection(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = f.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_projection(output)
        return output




