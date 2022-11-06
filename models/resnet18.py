# @Author: Enea Duka
# @Date: 3/7/22
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


def create_resnet18(model_pretrain_path=None):
    model = ResNet18()

    if model_pretrain_path:
        saved_model = torch.load(model_pretrain_path)
        state_dict = saved_model["state_dict"]
        model.backbone.load_state_dict(state_dict, strict=False)

    model.cuda()
    model = nn.DataParallel(model)

    return model
