import os
import json
import torch
import argparse
import matplotlib.pyplot as plt

#from loss import Loss
from models.vq import get_model 
from loss import Loss
from dataloader import ShapesDataset
from train import train
from plotting import save_loss

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='dev')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vis_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lat_dim', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


# save args to .json
def save_args(args):
    save_path = os.path.join('results', args.test_name, 'args.json')
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

# TODO: make config file for dataset
if __name__ == '__main__':
    # get args
    args = get_args()
    exp_path = os.path.join('results', args.exp_name)
    os.makedirs(exp_path, exist_ok=True)

    # dataloader 
    print('loading dataset')
    x = torch.load('data/imgs.pt')
    y = torch.load('data/labels.pt')
    print(x.shape) 
    print(y.shape)
    quit()

    x = x.permute(0, 3, 1, 2) / 255
    loader = ShapesDataset(x, y, args, shuffle=True)

    # get model and optimizer
    # size, lat_dim, channels
    vae = VAE(x.shape[2], args.lat_dim).to(args.device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # loss function
    loss_fn = Loss(args)

    loss_track = []; best_loss = np.inf
    for epoch in range(args.epochs):
        avg_loss = train(vae, loader, loss_fn, opt, args)
        loss_track.append(avg_loss)

        # update best loss and save model
        if avg_loss[2] < best_loss:
            best_loss = avg_loss[2]
            torch.save(vae.state_dict(), os.path.join(args.name, 'best_model.pth'))

        # save and print loss
        print(f'Epoch: {epoch+1}/{args.epochs}, Loss: {avg_loss[2]:.6f}, Recon: {avg_loss[0]:.6f}, KLD: {avg_loss[1]:.4f}')
        save_loss(loss_track, path=args.name)
