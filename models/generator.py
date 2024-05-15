"""Unet-based generator"""

import torch as T
from torch import nn


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, num_downs, nf=64):
        super(Generator, self).__init__()
        # construct unet block
        unet_block = UnetBlock(nf * 8, nf * 8, in_dim=None, submodule=None, innermost=True) 
        for _ in range(num_downs - 5):          
            unet_block = UnetBlock(nf * 8, nf * 8, in_dim=None, submodule=unet_block)
        
        unet_block = UnetBlock(nf * 4, nf * 8, in_dim=None, submodule=unet_block)
        unet_block = UnetBlock(nf * 2, nf * 4, in_dim=None, submodule=unet_block)
        unet_block = UnetBlock(nf, nf * 2, in_dim=None, submodule=unet_block)
        self.model = UnetBlock(out_dim, nf, in_dim=in_dim, submodule=unet_block, outermost=True)  

    def forward(self, input):
        return self.model(input)


class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, out_dim, inner_dim, in_dim=None,
                 submodule=None, outermost=False, innermost=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if in_dim is None:
            in_dim = out_dim
        downconv = nn.Conv2d(in_dim, inner_dim, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.InstanceNorm2d(inner_dim)
        uprelu = nn.ReLU(True)
        upnorm = nn.InstanceNorm2d(out_dim)
        upconv = nn.ConvTranspose2d(inner_dim * 2, out_dim,
                                        kernel_size=4, stride=2,
                                        padding=1)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_dim * 2, out_dim,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_dim, out_dim,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_dim * 2, out_dim,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   
            return T.cat([x, self.model(x)], 1)



class UnetBlockWithZ(nn.Module):
    def __init__(self, in_dim, out_dim, inner_dim, z_dim, submodule, 
                 outermost=False, innermost=False, norm_layer=None):
        super(UnetBlockWithZ, self).__init__()
        
        self.outermost = outermost
        self.innermost = innermost
        in_dim = in_dim + z_dim
        downconv = [nn.Conv2d(in_dim, inner_dim,
                               kernel_size=4, stride=2, padding=1)]
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(inplace=True)

        if outermost:
            down = downconv
            up = [nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(inner_dim * 2, out_dim, 
                                     kernel_size=4, stride=2, padding=1),
                  nn.Tanh()]
        elif innermost:
            down = [downrelu] + downconv
            up = [nn.ReLU(inplace=True), 
                  nn.ConvTranspose2d(inner_dim, out_dim, kernel_size=4,
                                      stride=2, padding=1)] 
            if norm_layer is not None:
                up += [norm_layer(out_dim)]
        else:
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_dim)]
            upconv = [nn.ConvTranspose2d(
            inner_dim * 2, out_dim, kernel_size=4, stride=2, padding=1)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(out_dim)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        z_img = z.view(z.size(0), z.size(1), 1, 1) \
            .expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z = T.cat([x, z_img], 1)
        
        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return T.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return T.cat([self.up(x2), x], 1)
