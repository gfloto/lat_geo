import math
import torch
import torch.nn as nn

# calculate output shape of convolution
def output_shape(in_shape, kernel_size, stride, padding, dilation=1):
    return math.floor(in_shape + 2*padding - dilation*(kernel_size-1) - 1 / stride + 1)

def padding_required(in_shape, out_shape, kernel_size, stride, dilation=1):
    return math.ceil(((out_shape - 1)*stride + dilation*(kernel_size-1) - in_shape + 1)/2)

# squeeze and excitation block
class SE(nn.Module):
    def __init__(self, in_c, reduction=16):
        super(SE, self).__init__()

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // reduction, in_c, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
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

# final linear layers
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
            nn.Linear(self.lin_size // 2, lat_dim)
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