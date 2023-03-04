import os, sys
import torch
import numpy as np

from plotting import save_vis

def train(model, loader, opt, args):
    loss_track = []
    for i, (x, y) in enumerate(loader):
        x = x.to(args.device); y = y.to(args.device)

        # forward pass
        opt.zero_grad()
        mu, z, x_out = model(x)
        recon, kld, loss = model.loss(x, x_out, mu)
        loss_track.append([recon.cpu().item(), kld.cpu().item(), loss.cpu().item()])

        # backward pass
        loss.backward()
        opt.step()

        # visualize
        if i % args.vis_freq == 0:
            save_vis(x, x_out, path=args.name)

    # get average loss
    loss_track = np.array(loss_track)
    avg_loss = np.mean(loss_track, axis=0)
    return avg_loss
