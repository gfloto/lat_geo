import sys, os, time
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from loss import Loss
from model import VAE, Encoder, Decoder
from dataloader import ShapesDataset
from train import train
from plotting import save_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='dev')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--vis_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

# TODO: run this is half precision and check speed...
if __name__ == '__main__':
    # get args
    args = get_args()
    args.name = os.path.join('results', args.name)
    os.makedirs(args.name, exist_ok=True)

    # get device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader 
    x = torch.tensor(np.load('data/3dshapes_imgs.npy'), dtype=torch.float32).to(args.device)
    y = torch.tensor(np.load('data/3dshapes_labels.npy'), dtype=torch.float32).to(args.device)
    x = x.permute(0, 3, 1, 2)
    loader = ShapesDataset(x, y, args, shuffle=True)

    # get model and optimizer
    # size, lat_dim, channels
    vae = VAE(x.shape[2], y.shape[1]).to(args.device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    loss_track = []; best_loss = np.inf
    for epoch in range(args.epochs):
        avg_loss = train(vae, loader, opt, args)
        loss_track.append(avg_loss)

        # update best loss and save model
        if avg_loss[2] < best_loss:
            best_loss = avg_loss[2]
            torch.save(vae.state_dict(), os.path.join(args.name, 'best_model.pth'))

        # save and print loss
        print(f'Epoch: {epoch+1}/{args.epochs}, Loss: {avg_loss[2]:.6f}, Recon: {avg_loss[0]:.6f}, KLD: {avg_loss[1]:.4f}')
        save_loss(loss_track, path=args.name)