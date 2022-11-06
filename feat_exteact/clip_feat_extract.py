# @Author: Enea Duka
# @Date: 3/11/22
import sys

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")
import torch
import clip
import os
from tqdm import tqdm

from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
import numpy as np


if __name__ == "__main__":
    mode = "val"
    feature_base_path = ""
    feat_path = "%s/vit_feats/clip/%s_normalized/" % (feature_base_path, mode)
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)

    dataset = SimpleOopsDataset(
        mode,
        25,
        "frames",
        True,
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)
    model, preprocess = clip.load("RN50", device=device)
    model = torch.nn.DataParallel(model)
    print(str(model))
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            if "features" not in data.keys():
                continue
            frames = data["features"][0]
            array_path = os.path.join(
                feat_path, data["filename"][0].replace("mp4", "npz")
            )

            if frames.shape[0] > 128:
                outs = []
                frames = list(torch.split(frames, 128))
                for chunk in frames:
                    out = model.module.encode_image(chunk.cuda())
                    outs.append(out.cpu())
                output = torch.cat(outs, dim=0)
            else:
                output = model.module.encode_image(frames.cuda())
            p = 0.02
            sample = np.random.uniform(low=0.0, high=1.0, size=None)
            output_np = output.cpu().detach().numpy()

            np.savez_compressed(array_path, output_np)
