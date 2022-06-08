import torch
import torch.nn as nn

from core import ResidualConv, Upsample

padding_mode = 'reflect'


class ResUnet3D(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512], n_classes=1):
        super(ResUnet3D, self).__init__()

        self.n_classes = n_classes

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1, padding_mode=padding_mode),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1, padding_mode=padding_mode)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1, padding_mode=padding_mode)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1, padding_mode=padding_mode)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1, padding_mode=padding_mode)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1, padding_mode=padding_mode)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1, padding_mode=padding_mode)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1, padding_mode=padding_mode)

        self.output_layer = nn.Sequential(
            nn.Conv3d(filters[0], self.n_classes, 1, 1, padding_mode=padding_mode),
            # nn.Softmax(dim=32),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)
        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)
        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)
        # print(f'Output Shape: {output.shape}')
        return output
