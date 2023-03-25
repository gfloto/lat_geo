import sys, torch
import torch
from torch import nn
from torch.nn.functional import relu

# return positions of discrete latent variable maxima
def discrete_positions(args, sig=3):
    p = []
    for n in args.discrete:
        a = torch.linspace(0, sig*(n-1), n) - sig*(n-1)/2
        a = a.to(args.device)
        p.append(a.repeat(args.batch_size, 1).T)
    return p

# TODO: network is broken...
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.disc = args.discrete
        if args.discrete is not None:
            self.disc_pos = discrete_positions(args)

        self.lin = args.linear
        self.circ = args.circular
        self.lat_dim = args.lat_dim

        self.mu_save = None # for plotting histograms

    def forward(self, x, x_out, mu, ood=False):
        # l1 recon loss
        recon = torch.sum(torch.abs(x - x_out), dim=(1,2,3)).mean()

        # calculate each kld loss
        kld = 0
        if self.disc is not None:
            kld_disc = 0
            mu_disc = mu[:, :len(self.disc)]
            for i in range(len(self.disc)):
                dist = torch.abs(mu_disc[:, i] - self.disc_pos[i])
                m, _ = torch.min(dist, dim=0)
                kld_disc += m.mean()
            kld += kld_disc

        if self.lin > 0:
            mu_lin = mu[:, len(self.disc):len(self.disc)+self.lin]
            out = relu(mu_lin.abs() - 5)**2 # TODO: tune this
            kld_lin = out.sum(dim=1).mean()
            kld += kld_lin

        if self.circ > 0:
            print('circular not implemented yet')
            sys.exit()

        return recon, kld, recon + kld