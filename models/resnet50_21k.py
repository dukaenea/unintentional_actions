
# @Author: Enea Duka
# @Date: 2/6/22

import torch
import torch.nn as nn
from timm.models.resnet import resnet50


class ResNet5021K(nn.Module):

    def __init__(self):
        super(ResNet5021K, self).__init__()
        self.backbone = resnet50(pretrained=False, num_classes=0)

    def forward(self, x):
        return self.backbone(x)



def create_resnet50_21k(pretrained=False):
    model = ResNet5021K()

    if pretrained:
        saved_model = torch.load('/BS/unintentional_actions/nobackup/pretrained_models/resnet/resnet50_miil_21k.pth')
        state_dict = saved_model['state_dict']
        model.backbone.load_state_dict(state_dict, strict=False)

    model.cuda()
    model = nn.DataParallel(model)

    return model


if __name__ == '__main__':
    model = create_resnet50_21k(pretrained=True)
    print(model)