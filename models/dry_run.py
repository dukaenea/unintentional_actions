
# @Author: Enea Duka
# @Date: 4/22/21

import torch
from models.resnet50 import build_resnet_50
from models.r2plus1d import build_r2plus1d
from torchvision.models.video.resnet import VideoResNet
from models.vtn import VTN
from torchvision.models.resnet import resnet50

def dry_run_resnet50():
    # model_tv = resnet50(pretrained=False)
    model = build_resnet_50(pretrained=True)
    image = torch.randn((1, 3, 224, 224))

    out = model(image)
    print(out.shape)


def dry_run_r2plus1d():
    model, _, _ = build_r2plus1d()
    clip = torch.randn((1, 3, 32, 112, 112))

    out = model(clip)
    print(out.shape)

def dry_run_vtn():
    config = {}
    config['embed_dim'] = 768
    config['max_positions_embedding'] = 2 * 60 * 60
    config['num_attention_heads'] = 12
    config['num_hidden_layers'] = 3
    config['attention_mode'] = 'sliding_chunks'
    config['pad_token_id'] = -1
    config['attention_window'] = None
    config['intermediate_size'] = 3072
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['hidden_dim'] = 2048
    config['mlp_dropout'] = 0.2
    config['mlp_dim'] = 1024
    config['num_classes'] = 10

    model = VTN(config)
    print(model)


if __name__ == '__main__':
    # dry_run_resnet50()
    # dry_run_r2plus1d()
    dry_run_vtn()