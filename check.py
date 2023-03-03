from model import output_shape, padding_required

in_shape = 128
out_shape = 128
kernel_size = 3
stride = 1

pad = padding_required(in_shape, out_shape, kernel_size, stride)
out = output_shape(in_shape, kernel_size, stride, pad)

# print info
print('input shape: {}'.format(in_shape))
print('output shape: {}'.format(out))
print('kernel size: {}'.format(kernel_size))
print('stride: {}'.format(stride))
print(20*'-')
print('padding: {}'.format(pad))