import torch
import torch.nn as nn
from torchsummary import summary


class DiffusionModel(nn.Module):
    def __init__(self, input_channels=3, num_blocks=5, base_channels=64):
        super(DiffusionModel, self).__init__()
        self.num_blocks = num_blocks
        self.base_channels = base_channels

        # Define the convolutional blocks
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.conv_blocks.append(self._build_conv_block(input_channels))
            input_channels *= 2

        # Define the final output layer
        self.final_conv = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)

    def _build_conv_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            skips.append(x)
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = torch.cat([x] + skips[::-1], dim=1)
        x = self.final_conv(x)
        return x


# Move model and input data to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel().to(device)
input_tensor = torch.randn(1, 3, 256, 256).to(device)

# Summarize the model
summary(model, (3, 256, 256))
