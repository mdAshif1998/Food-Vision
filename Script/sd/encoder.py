import torch
from torch import nn
from torch.nn import functional as f
from decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(

            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=(3, 3), padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAEResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAEResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=0),

            # (Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAEResidualBlock(128, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAEResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=0),

            # (Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAEResidualBlock(256, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAEResidualBlock(512, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=0),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAEResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAEResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAEResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAEAttentionBlock(512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAEResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.SiLU(),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=(3, 3), padding=1),

            # (Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=(1, 1), padding=1),

        )

    def forward(self, *val: [torch.Tensor]) -> torch.Tensor:
        x = val[0]
        noise = val[1]
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = f.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (Batch_Size, 4, Height / 8, Width / 8) ->  (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (Batch_Size, 4, Height / 8, Width / 8) ->  (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()

        # (Batch_Size, 4, Height / 8, Width / 8) ->  (Batch_Size, 4, Height / 8, Width / 8)
        std = variance.sqrt()

        # Z = N(0, 1) -> N(mean, variance)=x?
        # X = mean + std * Z

        x = mean + std * noise

        # Scale the output by a constant
        x *= 0.18215

        return x

