# @Author: Enea Duka
# @Date: 8/23/21

import torch
from scipy.stats.distributions import norm
from scipy.stats import multivariate_normal
import numpy as np

if __name__ == '__main__':

    mean = torch.zeros((128,))
    std = torch.ones((128,))

    mean = torch.tensor(mean)
    std = torch.tensor(std)

    train_samples = []
    for i in range(1000):
        train_samples.append(torch.normal(mean, std))

    train_samples = torch.stack(train_samples)
    print(train_samples)
    train_samples = train_samples.cpu().numpy()

    _mean = np.average(train_samples, axis=0)
    _cov = np.cov(train_samples, rowvar=False)
    print(np.linalg.det(_cov))

    prob = multivariate_normal(_mean, _cov)

    in_dist = torch.normal(mean, std)
    in_p = prob.pdf(in_dist)
    print(in_p)

    out_dist = torch.normal(mean, 4*std)
    out_p = prob.pdf(out_dist)
    print(out_p)

    print('Checks out') if in_p > out_p else print('Doesn\'t check out')


