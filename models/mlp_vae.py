# @Author: Enea Duka
# @Date: 8/23/21

import torch
import torch.nn as nn
from utils.arg_parse import opt


class MLP_AE(nn.Module):
    def __init__(self, in_dim, out_dim, variational=False):
        super(MLP_AE, self).__init__()

        self.variational = variational

        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            # nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(out_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim)
        )

        if self.variational:
            self.mean_ll = nn.Linear(out_dim, out_dim)
            self.var_ll = nn.Linear(out_dim, out_dim)

    def forward(self, x, return_latent=False):
        x = self.encoder(x)
        if self.variational:
            mu = self.mean_ll(x)
            logvar = self.var_ll(x)
            x = self.reparametrize(mu, logvar)

        if return_latent:
            if self.variational:
                return x, mu, torch.exp(0.5*logvar)
            else:
                return x

        if self.variational:
            return self.decoder(x), mu, logvar
        else:
            return self.decoder(x)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)

        return mu + std*eps

    def kl_div(self, mu, sigma):
        sigma = torch.exp(0.5*sigma)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma)-1/2).sum()

def create_model(in_dim, out_dim):
    model = MLP_AE(in_dim, out_dim)
    model.cuda()
    model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    loss = nn.MSELoss(reduction='sum')

    return model, optimizer, loss
