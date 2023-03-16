import numpy as np
from model import output_shape, padding_required
from model import output_shape_T, pads_required_T

def check(size, kernel_size=3, stride=1):
    pad = padding_required(size, size, kernel_size, stride)
    out = output_shape(size, kernel_size, stride, pad)
    assert size == out, f'conv: expected {size} got {out}'

def check_T(in_size, out_size, kernel_size=3, stride=1):
    pad, pad_out = pads_required_T(in_size, out_size, kernel_size, stride)
    out = output_shape_T(in_size, kernel_size, stride, pad, output_padding=pad_out)
    print(f'pad: {pad}, pad_out: {pad_out}, out: {out}, out_size: {out_size}')
    assert out_size == out, f'conv_T expected {out_size} got {out}, where in_size: {in_size}, pad: {pad}'

if __name__ == '__main__':
    for _ in range(1000):
        n = np.random.randint(1000)
        print(f'testing: {n//2} -> {n}')
        check_T(n//2, n)
    print('passed')
