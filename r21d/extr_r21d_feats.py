# @Author: Enea Duka
# @Date: 11/11/21
import sys

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")

from r21d_model import r2plus1d_18
from dataloaders.oops_loader import get_video_loader_frames
from utils.arg_parse import opt
from tqdm import tqdm
import os
import uuid
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0 1'
    mode = "val"
    base_feat_path = os.path.join("../resources/data/features/r21d_features", mode)

    model = r2plus1d_18(pretrained=True, progress=True)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    opt.batch_size = 16
    opt.workers = 32
    opt.balance_fails_only = True
    opt.all_fail_videos = False
    opt.load_videos = False
    opt.step_between_clips_sec = 0.25
    if mode == "train":
        loader = get_video_loader_frames(opt)
    else:
        opt.val = True
        opt.fails_path = ""
        loader = get_video_loader_frames(opt)

    for idx, data in enumerate(tqdm(loader)):

        videos = data["features"]
        labels = data["label"]
        video_names = data["video_name"]
        clip_idc = data["clip_idx"]

        try:
            out = model(videos)
        except Exception as e:
            print(e)

        for i, o in enumerate(out):
            out_dict = {"feature": o.detach().cpu().numpy(), "label": labels[i].item()}
            file_path = (
                os.path.join(base_feat_path, video_names[i]) + "~%d.npy" % clip_idc[i]
            )

            if os.path.isfile(file_path):
                print("file_exists")

            np.save(file_path, out_dict)
