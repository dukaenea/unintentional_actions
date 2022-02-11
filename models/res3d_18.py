# @Author: Enea Duka
# @Date: 1/26/22

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from utils.arg_parse import opt
from utils.logging_setup import logger


class R3D18(nn.Module):

    def __init__(self, pretrain_backbone=True, local=False):
        super(R3D18, self).__init__()

        self.backbone = r3d_18(pretrained=pretrain_backbone)

        self.backbone.fc = nn.Sequential(
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            nn.BatchNorm1d(opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(opt.mlp_dim, opt.num_classes if opt.num_classes_ptr is None else opt.num_classes_ptr),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

    def reinit_mlp(self):
        self.backbone.fc = nn.Sequential(
            nn.LayerNorm(opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(opt.mlp_dim, opt.num_classes),
        )

    def remove_mlp(self):
        self.backbone.fc = IdentityLayer()


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

def create_r3d(pretrained):
    model = R3D18(pretrain_backbone=False)
    model.cuda()

    if opt.gpu_parallel:
        model = nn.DataParallel(model)

        # if we have a model that we pretrained, then we need to reinit the mlp
        if pretrained:
            saved_model = torch.load(opt.resnet_ptr_path)
            model_dict = saved_model['state_dict']
            model.load_state_dict(model_dict, strict=True)
            model.module.reinit_mlp()
            model.module.cuda()
            # if opt.gpu_parallel:
            #     model = nn.parallel.DistributedDataParallel(model.module)

    if opt.optim == 'adam':
        if opt.rep_learning:
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    loss = nn.CrossEntropyLoss()

    logger.debug(str(model))
    logger.debug(str(optimizer))
    logger.debug(str(loss))
    return model, optimizer, loss


if __name__ == '__main__':
    print('hello')