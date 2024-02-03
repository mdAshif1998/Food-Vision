import torch
from torch import nn
from torch.nn import functional as f
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3, 320)

        # x: (3, 320) -> (1, 1280)
        x = self.linear_1(x)
        # (1, 1280) -> (1, 1280)
        x = f.selu(x)
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)
        # (1, 1280)
        return x


class UNETResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.group_norm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3 ), padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)

    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)
        residue = feature
        feature = self.group_norm_feature(feature)

        feature = f.silu(feature)
        feature = self.conv_feature(feature)

        time = f.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.group_norm_merged(merged)
        merged = f.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNETAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super().__init__()
        channels = n_head * n_embd

        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_projection_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_projection_bias=False)
        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_geg_lu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geg_lu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)

    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        residue_long = x

        x = self.group_norm(x)
        x= self.conv_input(x)
        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) - > (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection

        residue_short = x

        x = self.layer_norm_1(x)
        self.attention_1(x)
        x += residue_short

        residue_short = x

        # Normalization + Cross Attention with skip connection
        x = self.layer_norm_2(x)
        self.attention_2(x, context)

        x += residue_short

        residue_short = x

        # Normalization + Feed Forward Layer with GeGLU activation function and skip connection
        x = self.layer_norm_3(x)

        x, gate = self.linear_geg_lu_1(x).chunk(2, dim=1)
        x = x * f.gelu(gate)
        x = self.linear_geg_lu_2(x)
        x += residue_short

        # (Batch_Size, Height * Width, Features) - > (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        return self.conv_output(x) + residue_long


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = f.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class SwitchSequential(nn.Sequential):

    def forward(self, *val: [torch.Tensor]) -> torch.Tensor:
        x = val[0]
        assert isinstance(x, torch.Tensor)
        context = val[1]
        assert isinstance(context, torch.Tensor)
        time = val[2]
        assert isinstance(time, torch.Tensor)
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=(3, 3), padding=1)),

            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),

            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=1)),

            SwitchSequential(UNETResidualBlock(320, 640), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBlock(640, 640), UNETAttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=1)),

            SwitchSequential(UNETResidualBlock(640, 1280), UNETAttentionBlock(8, 160)),

            SwitchSequential(UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=1)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(1280, 1280))

        ])

        self.bottleneck = SwitchSequential(

            UNETResidualBlock(1280, 1280),

            UNETAttentionBlock(8, 160),

            UNETResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(2560, 1280)),

            SwitchSequential(UNETResidualBlock(2560, 1280)),

            SwitchSequential(UNETResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),

            SwitchSequential(UNETResidualBlock(1920, 1280), UNETAttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBlock(1280, 640), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBlock(960, 640), UNETAttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNETResidualBlock(960, 320), UNETAttentionBlock(8, 40)),

            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),

        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # (Batch_Size, 320, Height / 8, Width / 8)
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.group_norm(x)
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = f.selu(x)
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Latent is output of the Encoder, Prompt/context is the output of CLIP, Time is an embedding like for which timestep the Latent needs to
        # De-noisify by the UNET
        # Latent: (Batch_Size, 4, Height / 8, Width / 8)
        # Context: (Batch_Size, Seq_Len, Dim-> 768)
        # Time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 4, Width / 4) -> (Batch, 320, Height / 4, Width / 4)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 4, Width / 4) - > (Batch, 4, Height / 4, Width / 4)
        output = self.final(output)

        # Latent: (Batch, 4, Height / 4, Width / 4)
        return output








