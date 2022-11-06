# @Author: Enea Duka
# @Date: 4/24/21

import torch
from tqdm import tqdm
import os
import numpy as np
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.rareacts_loader import RareactsDataset
from torch.utils.data import DataLoader

from models.r2plus1d import build_r2plus1d


def extract_r2plus1d_feats(dataset_name, mode, base_feature_path):
    feat_path = "%s/%s/r2plus1d_feats/%s_f32/" % (base_feature_path, dataset_name, mode)
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    if dataset_name == "kinetics":
        # use the ImageNet stats
        dataset = KineticsDataset(
            mode,
            25,
            False,
            False,
            {"mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]},
            fpc=32,
            feat_ext=True,
        )
    elif dataset_name == "rareact":
        dataset = RareactsDataset(
            mode,
            25,
            False,
            False,
            {"mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]},
        )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=32, shuffle=False)

    model, _, _ = build_r2plus1d()
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            frames = data["features"]

            # use the number of clips as the batch size to feed to r2+1d
            output = model(frames[0].cuda())
            # if idx % 10 != 0:
            #     output = output.mean(4).mean(3)
            output_np = frames.squeeze().cpu().detach().numpy()
            # # float64 consumes a lot of space, using float16 to save space
            # output_np = output_np.astype(np.float16)
            label_folder_path = os.path.join(feat_path, data["label_text"][0])
            if not os.path.exists(label_folder_path):
                os.makedirs(label_folder_path)
            array_path = os.path.join(label_folder_path, data["video_name"][0]) + ".npz"
            #
            np.savez_compressed(array_path, output_np)
            # torch.save(output, array_path)
            #
            # if idx == 10:
            #     break


if __name__ == "__main__":
    extract_r2plus1d_feats("kinetics", "val", "")
