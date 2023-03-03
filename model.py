import math
import torch
import torch.nn as nn

# calculate output shape of convolution
def output_shape(in_shape, kernel_size, stride, padding, dilation=1):
    return math.floor(in_shape + 2*padding - dilation*(kernel_size-1) - 1 / stride + 1)

def padding_required(in_shape, out_shape, kernel_size, stride, dilation=1):
    return math.ceil(((out_shape - 1)*stride + dilation*(kernel_size-1) - in_shape + 1)/2)

class squeeze_excite_block(nn.Module):
    def __init__(self, in_c, reduction=16):
        super(squeeze_excite_block, self).__init__()

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // reduction, in_c, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.block(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            squeeze_excite_block(32),
        )

        
    def forward(self, x):
        x = self.block(x)
        print(x.shape)
        return x