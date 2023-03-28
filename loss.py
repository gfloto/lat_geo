import sys, torch
import torch
from torch import nn
from torch.nn.functional import relu

# return positions of discrete latent variable maxima
def discrete_positions(args, sig):
    p = []
    for n in args.discrete:
        a = torch.linspace(0, sig*(n-1), n) - sig*(n-1)/2
        a = a.to(args.device)
        #p.append(a.repeat(args.batch_size, 1).T)
    return a 

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.disc = args.discrete
        if args.discrete is not None:
            self.disc_sig = 6
            self.disc_pos = discrete_positions(args, sig=self.disc_sig)

        self.lin = args.linear
        self.circ = args.circular
        self.lat_dim = args.lat_dim

        self.mu_save = None # for plotting histograms

    # NOTE: using labels is tied to the dataloader format!
    def forward(self, x, x_out, y, mu, ood=False):
        # l1 recon loss
        recon = torch.sum(torch.abs(x - x_out), dim=(1,2,3)).mean()

        # calculate each kld loss
        # TODO: for loops are slow and for noobs (get better kid): use mask strategy instead
        kld = 0
        if self.disc is not None :
            kld_disc = 0
            mu_disc = mu[:, :len(self.disc)]

            # iterate over each discrete latent variable and batch
            for j in range(len(self.disc)):
                temp_kld = 0
                for i in range(mu_disc.shape[0]):
                    # no label
                    if torch.isnan(y[i,j]).any():
                        dist = (mu_disc[i,j] - self.disc_pos)**2
                        m, _ = torch.min(dist, dim=0)
                        temp_kld += m.mean()
                    
                    # pin latent to label location
                    else:
                        temp_kld += (mu_disc[i,j] - self.disc_pos[int(y[i,j])])**2

                # get average kld for discrete latent variables
                kld_disc += temp_kld / mu_disc.shape[0]
            kld += kld_disc

        if self.lin > 0:
            mu_lin = mu[:, len(self.disc) : len(self.disc)+self.lin]
            y_lin = y[:, len(self.disc) : len(self.disc)+self.lin]
            mask = ~torch.isnan(y_lin) # 0 is nan,1 otherwise
            mask = mask.float()

            # label
            label_kld = torch.nan_to_num((mu_lin - y_lin)**2)

            # no label kld
            # TODO: tune this
            nolabel_kld = relu(mu_lin.abs() - 10)**2
            kld_lin = (1-mask) * nolabel_kld + mask * label_kld
            kld += kld_lin.sum(dim=1).mean()

        if self.circ > 0:
            print('circular not implemented yet')
            sys.exit()

        return recon, kld, recon + kld