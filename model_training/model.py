import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride, padding=kernel_size//2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class WaveUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, depth=5, base_ch=24):
        super(WaveUNet, self).__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            self.encoders.append(ConvBlock(ch, base_ch * (2 ** i)))
            ch = base_ch * (2 ** i)

        self.bottleneck = ConvBlock(ch, ch)

        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in reversed(range(depth)):
            self.upsamplers.append(nn.ConvTranspose1d(ch, base_ch * (2 ** i), kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(base_ch * (2 ** i) * 2, base_ch * (2 ** i)))
            ch = base_ch * (2 ** i)

        self.final = nn.Conv1d(ch, out_ch, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = F.avg_pool1d(x, kernel_size=2)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skip_connections)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1])
            x = torch.cat((x, skip), dim=1)
            x = decoder(x)

        return self.final(x)