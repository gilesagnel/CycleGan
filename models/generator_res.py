"""
Resent based generator which has residual blocks between downsampling and up sampling operation
Take idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
"""
from torch import nn

class ResnetGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, nf=64, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        self.padding_type = 'reflect'
        self.model = self.build_model(in_dim, out_dim, nf, n_blocks)


    def build_model(self, in_dim, out_dim, nf, n_blocks):
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            ]

        def deconv_block(in_channels, out_channels):
            return [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            ]

        model = [
            nn.ReflectionPad2d(3),
            *conv_block(in_dim, nf, kernel_size=7, padding=0),
            *conv_block(nf, nf * 2, stride=2),
            *conv_block(nf * 2, nf * 4, stride=2)
        ]

        for _ in range(n_blocks):
            model += [ResnetBlock(nf * 4, padding_type=self.padding_type)]

        model += [
            *deconv_block(nf * 4, nf * 2),
            *deconv_block(nf * 2, nf),
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, out_dim, kernel_size=7),
            nn.Tanh()
        ]

        return nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Residual block"""

    def __init__(self, dim, padding_type):
        super(ResnetBlock, self).__init__()
        padding_layer = self.get_padding_layer(padding_type)
        conv_block = [
            padding_layer(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            padding_layer(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def get_padding_layer(self, padding_type):
        if padding_type == 'reflect':
            return nn.ReflectionPad2d
        elif padding_type == 'replicate':
            return nn.ReplicationPad2d
        elif padding_type == 'zero':
            return lambda pad: nn.ZeroPad2d(pad)

    def forward(self, x):
        out = x + self.conv_block(x)  
        return out


if __name__ == "__main__":
    g = ResnetGenerator(3, 3, 64, 9)
    print(g)