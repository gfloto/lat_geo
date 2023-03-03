import sys, math
import torch
import torch.nn as nn

'''
the encoder and decoder are roughly based on the following paper:
https://arxiv.org/abs/2007.03898

batch norm is not included in this implimentation due to potential statistical issues
'''

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
        x = x + self.block(x)
        return x

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
    def __init__(self, size, lat_dim):
        super(Encoder, self).__init__()        
        out_c = 32
        assert math.log(size, 2) % 1 == 0, 'size must be a power of 2'
        self.layers = int(math.log(size, 2) - 3) # shrink until bxcx8x8 then linear -> bxlat_dim
        self.sizes = [size // 2**i for i in range(self.layers+1)]

        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            in_c = 32 if i != 0 else 3
            self.blocks.append(EncShrink(self.sizes[i], self.sizes[i+1], in_c, out_c))
        self.linear = EncLinear(lat_dim, self.sizes[-1], in_c, in_c // 2)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x) 
        x = self.linear(x)
        return x

# final linear layers for decoder
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
        print(x.shape)
        x = self.block(x)
        x = x.view(-1, self.in_c, self.size, self.size)
        return self.conv(x)

# decoder model
class Decoder(nn.Module):
    def __init__(self, size, lat_dim):
        super(Decoder, self).__init__()
        out_c = 32
        assert math.log(size, 2) % 1 == 0, 'size must be a power of 2'
        self.layers = int(math.log(size, 2) - 3) # shrink until bxcx8x8 then linear -> bxlat_dim
        self.sizes = [size // 2**i for i in range(self.layers+1)]
        self.sizes.reverse()

        self.linear = DecLinear(lat_dim, 8, 16, 32)
        #self.blocks = nn.ModuleList()
        #for i in range(self.layers):
            #in_c = 32 if i != 0 else 3
            #self.blocks.append(EncShrink(self.sizes[i], self.sizes[i+1], in_c, out_c))
        #self.final = nn.Conv2d(out_c, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.linear(x)
        print(x.shape)
        #x = x.view(-1, 32, self.sizes[-1], self.sizes[-1])
        #for block in self.blocks:
            #x = block(x)
        #x = self.final(x)
        #return x