# @Author: Enea Duka
# @Date: 1/26/22

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from utils.arg_parse import opt
from utils.logging_setup import logger
from torchcrf import CRF
from itertools import chain


class R3D18(nn.Module):
    def __init__(self, pretrain_backbone=True, local=False):
        super(R3D18, self).__init__()

        self.backbone = r3d_18(pretrained=pretrain_backbone)
        self.backbone.fc = IdentityLayer()

        if opt.use_crf:
            self.crf = CRF(num_tags=3, batch_first=True)

    def forward(self, x, labels=None, for_crf=False):
        if for_crf:
            # create the mask for the CRF layer
            crf_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.uint8).to(
                x.device
            )
            mask = (x == 0)[:, :, 0]
            crf_mask[mask] = 0
            if self.training:
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(0)
                loss = -self.crf.forward(x, labels, mask=crf_mask, reduction="mean")
                return loss
            else:
                loss = -self.crf.forward(x, labels, mask=crf_mask, reduction="mean")
                mls = self.crf.decode(x, mask=crf_mask)
                return torch.tensor(list(chain.from_iterable(mls))).to(x.device), loss

        x = self.backbone(x)
        return x

    def reinit_mlp(self):
        self.backbone.fc = nn.Sequential(
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            nn.BatchNorm1d(opt.mlp_dim),
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
            model.module.reinit_mlp()
            saved_model = torch.load(opt.resnet_ptr_path)
            model_dict = saved_model["state_dict"]
            model.load_state_dict(model_dict, strict=True)
            model.module.cuda()
            if opt.gpu_parallel:
                model = nn.parallel.DistributedDataParallel(model.module)

    if opt.optim == "adam":
        if opt.rep_learning:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            )
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )

    loss = nn.CrossEntropyLoss()
    # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([19670 / 18069, 19670 / 4137, 19670 / 19670]).cuda())

    logger.debug(str(model))
    logger.debug(str(optimizer))
    logger.debug(str(loss))
    return model, optimizer, loss


if __name__ == "__main__":
    print("hello")
