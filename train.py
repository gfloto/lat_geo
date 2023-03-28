import os, sys
import torch
import numpy as np

from plotting import save_vis

def train(model, loader, loss_fn, opt, args):
    loss_track = []
    for i in range(loader.n_batch):
        if i > 1000: break
        x, y = loader.get_batch(i)
        x = x.to(args.device)
        y = y.to(args.device)

        # forward pass
        opt.zero_grad()
        mu, z, x_out = model(x)
        recon, kld, loss = loss_fn(x, x_out, y, mu)
        loss_track.append([recon.cpu().item(), kld.cpu().item(), loss.cpu().item()])

        # backward pass
        loss.backward()
        opt.step()

        # visualize and print loss
        if i % args.vis_freq == 0:
            lt = np.array(loss_track)
            print(f'Recon: {np.mean(lt[:, 0])}, KLD: {np.mean(lt[:, 1])}')
            save_vis(x, x_out, path=args.name)

    # get average loss
    loss_track = np.array(loss_track)
    avg_loss = np.mean(loss_track, axis=0)
    return avg_loss
