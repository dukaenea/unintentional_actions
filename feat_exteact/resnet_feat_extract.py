# @Author: Enea Duka
# @Date: 4/22/21

import torch
from tqdm import tqdm
import os
import numpy as np
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.rareacts_loader import RareactsDataset
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader

from models.resnet18 import create_resnet18


def extract_resnet_features(dataset_name, mode, base_feature_path, model_pretrain_path):
    feat_path = "%s/%s/resnet_feats/resnet_18/%s_normalized/" % (
        base_feature_path,
        dataset_name,
        mode,
    )
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)

    if dataset_name == "kinetics":
        # use the ImageNet stats
        dataset = KineticsDataset(
            mode,
            25,
            True,
            False,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            feat_ext=True,
        )
    elif dataset_name == "rareact":
        dataset = RareactsDataset(
            mode,
            25,
            False,
            False,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
    elif dataset_name == "oops":
        dataset = SimpleOopsDataset(
            mode,
            25,
            "frames",
            True,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    model = create_resnet18(model_pretrain_path)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            if "features" not in data.keys():
                continue
            frames = data["features"][0]
            array_path = os.path.join(
                feat_path, data["filename"][0].replace("mp4", "npz")
            )
            output = model(frames.cuda())
            output_np = output.cpu().detach().numpy()

            np.savez_compressed(array_path, output_np)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    extract_resnet_features("oops", "val", "", None)
