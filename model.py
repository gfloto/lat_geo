import sys, math
import torch
import torch.nn as nn

'''
the encoder and decoder are roughly based on the following paper:
https://arxiv.org/abs/2007.03898

batch norm is not included in this implimentation due to potential statistical issues
'''
# TODO: make conv h/w before linear more general

# vae
class VAE(nn.Module):
    def __init__(self, size, lat_dim, channels=32):
        super(VAE, self).__init__()
        self.enc = Encoder(size, lat_dim, channels)
        self.dec = Decoder(size, lat_dim, channels)

        # print model params
        print(f'enc params: {sum(p.numel() for p in self.enc.parameters() if p.requires_grad):,}') 
        print(f'dec params: {sum(p.numel() for p in self.dec.parameters() if p.requires_grad):,}')

    def loss(self, x, x_out, mu, r=10):
        recon = torch.mean(torch.sum((x - x_out)**2, dim=(1,2,3)))
        kld = torch.mean((mu.norm(dim=1) - r)**2)
        loss = recon + kld
        return recon, kld, loss

    def forward(self, x):
        mu = self.enc(x)
        z = mu + torch.randn_like(mu)
        return mu, z, self.dec(z)

# calculate output shape of convolution
def output_shape(in_shape, kernel_size, stride, padding, dilation=1):
    return math.floor(in_shape + 2*padding - dilation*(kernel_size-1) - 1 / stride + 1)

# calculate padding required for convolution
def padding_required(in_shape, out_shape, kernel_size, stride, dilation=1):
    return math.ceil(((out_shape - 1)*stride + dilation*(kernel_size-1) - in_shape + 1) / 2)

# calculate output shape of transposed convolution
def output_shape_T(in_shape, kernel_size, stride, padding, dilation=1, output_padding=0):
    return math.floor((in_shape - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1)

# calculate padding required for transposed convolution
def padding_required_T(in_shape, out_shape, kernel_size, stride, dilation=1, output_padding=0):
    return math.floor(((in_shape - 1)*stride + dilation*(kernel_size-1) + 1 + output_padding + 1 - out_shape) / 2)

# squeeze and excitation block
class SE(nn.Module):
    def __init__(self, in_c, reduction=16):
        super(SE, self).__init__()

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // reduction, in_c, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

# TODO: clean up this implimentation
# NOTE: this shrinks by a factor of 2
# class to shrink and run resnet block
class EncShrink(nn.Module):
    def __init__(self, in_size, out_size, in_c, out_c):
        super(EncShrink, self).__init__()
        stride = 2
        kernel_size = 3
        pad = padding_required(in_size, out_size, kernel_size, stride)  

        self.shrink = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.ReLU(inplace=True),
        )

        stride = 1
        kernel_size = 3
        pad = padding_required(out_size, out_size, kernel_size, stride)   
        self.block = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=stride, padding=pad),
            SE(out_c),
        )

    def forward(self, x):
        x = self.shrink(x)
        return x + self.block(x)

# final linear layers for encoder
class EncLinear(nn.Module):
    def __init__(self, lat_dim, size, in_c, out_c):
        super(EncLinear, self).__init__()
        stride = 1
        kernel_size = 3
        pad = padding_required(size, size, kernel_size, stride)  

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.lin_size = out_c * size**2

        self.block = nn.Sequential(
            nn.Linear(self.lin_size, self.lin_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.lin_size // 2, lat_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.lin_size)
        return self.block(x)

# assume 
class Encoder(nn.Module):
    def __init__(self, size, lat_dim, c=32, b=4):
        super(Encoder, self).__init__()        
        assert math.log(size, 2) % 1 == 0, 'size must be a power of 2'
        self.layers = int(math.log(size, 4/3) - 7) # shrink until bxcx8x8 then linear -> bxlat_dim
        self.sizes = [int(size * (3/4)**i) for i in range(self.layers+1)]
        #self.layers = int(math.log(size, 2) - 3) # shrink until bxcx8x8 then linear -> bxlat_dim
        #self.sizes = [size // 2**i for i in range(self.layers+1)]
        self.sizes = [8, 12, 16, 24, 28, 36, 48, 64]
        self.sizes.reverse()
        self.layers = len(self.sizes)-1
        print(self.sizes, self.layers)

        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            for j in range(b):
                in_c = 3 if i == 0 and j == 0 else c
                self.blocks.append(EncShrink(self.sizes[i], self.sizes[i], in_c, c))
            self.blocks.append(EncShrink(self.sizes[i], self.sizes[i+1], in_c, c))
        self.linear = EncLinear(lat_dim, self.sizes[-1], c, c // 2)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x) 
        x = self.linear(x)
        return x

# Decoder
###########################################################################

# first linear layers for decoder
class DecLinear(nn.Module):
    def __init__(self, lat_dim, size, in_c, out_c):
        super(DecLinear, self).__init__()
        self.size = size
        self.in_c = in_c

        lin_size = in_c * size**2
        self.block = nn.Sequential(
            nn.Linear(lat_dim, lin_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(lin_size // 2, lin_size),
        )

        stride = 1
        kernel_size = 3
        pad = padding_required_T(size, size, kernel_size, stride)  

        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=pad)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, self.in_c, self.size, self.size)
        return self.conv(x)

# class to expand and run resnet block
class DecExpand(nn.Module):
    def __init__(self, in_size, out_size, in_c, out_c, e=4):
        super(DecExpand, self).__init__()
        stride = 1
        kernel_size = 3
        pad = padding_required_T(in_size, in_size, kernel_size, stride)  

        self.block = nn.Sequential(
            # 1x1 conv
            nn.Conv2d(in_c, e*in_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(e*in_c, e*in_c, kernel_size=kernel_size, stride=stride, padding=pad, groups=e*in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(e*in_c, in_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            SE(in_c),
        )

        stride = 2
        kernel_size = 3
        pad = padding_required_T(in_size, out_size, kernel_size, stride)  
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x + self.block(x)
        x = self.expand(x)
        return x

# decoder model
class Decoder(nn.Module):
    def __init__(self, size, lat_dim, c=32, b=4):
        super(Decoder, self).__init__()
        self.c = c
        assert math.log(size, 2) % 1 == 0, 'size must be a power of 2'
        # TODO: what is happening here???
        #self.layers = int(math.log(size, 2) - 3) # shrink until bxcx8x8 then linear -> bxlat_dim
        #self.sizes = [size // 2**i for i in range(self.layers+1)]
        #self.layers = int(math.log(size, 4/3) - 7) # shrink until bxcx8x8 then linear -> bxlat_dim
        #self.sizes = [int(size * (3/4)**i) for i in range(self.layers+1)]
        #self.sizes.reverse()
        #print(self.sizes, self.layers)

        #self.sizes = [8, 12, 16, 24, 28, 32, 36, 48, 64]
        self.sizes = [8, 12, 16, 24, 28, 36, 48, 64]
        self.layers = len(self.sizes) - 1
        print(self.sizes, self.layers)

        self.linear = DecLinear(lat_dim, 8, c//2, c)
        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            for j in range(b):
                self.blocks.append(DecExpand(self.sizes[i], self.sizes[i], c, c))
            self.blocks.append(DecExpand(self.sizes[i], self.sizes[i+1], c, c))
        
        stride = 1
        kernel_size = 3
        pad = padding_required(size, size, kernel_size, stride)  
        #self.final = nn.Conv2d(c, 256, kernel_size=kernel_size, stride=stride, padding=pad)

        self.final = nn.Sequential(
            nn.Conv2d(c, 3, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.Sigmoid(),
        )
        #self.final = nn.Conv2d(c, 256, kernel_size=kernel_size, stride=stride, padding=pad)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.c, 8, 8) 

        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x