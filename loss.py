import sys, torch
from torch import nn

class Loss(nn.Module):
    def __init__(self, discrete, linear, circular):
        super(Loss, self).__init__()
        self.disc = discrete
        self.lin = linear
        self.circ = circular
        self.nz = len(discrete) + linear + circular

    def forward(self, x, x_out, mu, logvar, ood=False):
        recon = torch.sum(torch.square(x - x_out), dim=(1,2,3))

        # kld loss options
        if self.tilt == 0.0:
            kld = -1/2 * (1 + logvar - mu.pow(2) - logvar.exp())
        else:
            mu_norm = torch.linalg.norm(mu, dim=1)
            kld = 1/2 * torch.square(mu_norm - self.mu_star)
        print(recon.shape, kld.shape)
        sys.exit()

        return recon, kld