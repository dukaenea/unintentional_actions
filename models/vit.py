# @Author: Enea Duka
# @Date: 6/20/21

from transformers import ViTModel, ViTFeatureExtractor
import torch.nn as nn
from PIL import Image
import torch
import numpy as np
from timm.models.vision_transformer import vit_base_patch16_224
from utils.arg_parse import opt

class ViTBackbone(nn.Module):

    def __init__(self):
        super(ViTBackbone, self).__init__()
        self.backbone = vit_base_patch16_224(pretrained=True,
                                             num_classes=0,
                                             drop_path_rate=0.1,
                                             drop_rate=0.1)

    def forward(self, x):
        out = self.backbone(x)
        # print(x.shape[0])
        return out


def create_vit_model(pretrain=False):
    model = ViTBackbone()
    model.cuda()
    model = nn.DataParallel(model)

    if pretrain:
        saved_model = torch.load(opt.vtn_ptr_path)
        model_dict = saved_model['vit_state_dict']
        model.load_state_dict(model_dict, strict=True)

    return model