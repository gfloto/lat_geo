import sys, os
import torch as pt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from loss import Loss
from model import Encoder, Decoder
from dataloader import celeba_dataset

# TODO: run this is half precision and check speed...
if __name__ == '__main__':
    batch_size = 32
    num_workers = 4
    size = 32

    # get model and lost
    loss = Loss(discrete=[2], linear=2, circular=1)

    # dataloader 
    loader = celeba_dataset(batch_size, num_workers, size)

    for i, (x, y) in enumerate(loader):
        print(x.shape, y.shape)
        if i > 10: break

