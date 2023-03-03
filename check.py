import numpy as np
from model import output_shape, padding_required
from model import output_shape_T, padding_required_T

def check(size, kernel_size=3, stride=1):
    pad = padding_required(size, size, kernel_size, stride)
    out = output_shape(size, kernel_size, stride, pad)
    assert size == out, f'conv: expected {size} got {out}'

def check_T(size, kernel_size=3, stride=1):
    pad = padding_required_T(size, size, kernel_size, stride)
    out = output_shape_T(size, kernel_size, stride, pad)
    assert size == out, f'conv_T expected {size} got {out}'

if __name__ == '__main__':
    for _ in range(1000):
        n = np.random.randint(10, 2000)
        check(n) 
        check_T(n) 
    print('passed')
