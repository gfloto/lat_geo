import sys, os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from loss import Loss
from model import Encoder
from dataloader import celeba_dataset

# TODO: run this is half precision and check speed...
if __name__ == '__main__':
    batch_size = 32
    num_workers = 4
    size = 128

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model and lost
    #loss = Loss(discrete=[2], linear=2, circular=1)
    model = Encoder().to(device)

    # dataloader 
    loader = celeba_dataset(batch_size, num_workers, size)

    for i, (x, y) in enumerate(loader):
        x = x.to(device); y = y.to(device)
        out = model(x) 
        print(x.shape, out.shape) 

