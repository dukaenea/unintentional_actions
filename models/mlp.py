# @Author: Enea Duka
# @Date: 5/20/21

import torch
import torch.nn as nn

from utils.arg_parse import opt


class MLP(nn.Module):
    def __init__(self, in_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_size, 2 * in_size), nn.GELU(), nn.Dropout(opt.dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2 * in_size, in_size // 2), nn.GELU(), nn.Dropout(opt.dropout)
        )
        self.layer3 = nn.Sequential(nn.Linear(in_size, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


def create_mlp_model(in_size, num_classes):
    model = MLP(in_size, num_classes)
    model.cuda()
    if opt.gpu_parallel:
        model = nn.DataParallel(model)

    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )
    if opt.optim == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    loss = nn.CrossEntropyLoss()

    return model, optimizer, loss
