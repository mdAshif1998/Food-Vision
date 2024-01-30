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

            weight.massked_fill(mask, -torch.inf)

            weight /= math.sqrt(self.d_heads)

            weight = f.softmax(weight, dim=-1)

            # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
            output = weight @ v

            # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
            output = output.transpose(1, 2)

            output = output.reshape(interim_shape)

            output = self.out_proj(output)

            # (Batch_Size, Seq_Len, Dim)
            return output


