import sys, os, time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from loss import Loss
from model import Encoder, Decoder
from dataloader import celeba_dataset

# TODO: run this is half precision and check speed...
if __name__ == '__main__':
    batch_size = 32
    num_workers = 4
    size = 64
    lat_dim = 10

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model and lost
    #loss = Loss(discrete=[2], linear=2, circular=1)
    enc = Encoder(size, lat_dim).to(device)
    dec = Decoder(size, lat_dim).to(device)

    # print model params
    print(f'enc params: {sum(p.numel() for p in enc.parameters() if p.requires_grad):,}') 
    print(f'dec params: {sum(p.numel() for p in dec.parameters() if p.requires_grad):,}')

    # dataloader 
    loader = celeba_dataset(batch_size, num_workers, size)

    for i, (x, y) in enumerate(loader):
        t0 = time.time()
        x = x.to(device); y = y.to(device)

        print(x.shape)
        z = enc(x) 
        x_out = dec(z)
        print(x_out.shape)

        print(f'time: {time.time() - t0:.5f}')
