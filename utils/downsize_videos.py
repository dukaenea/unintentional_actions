# @Author: Enea Duka
# @Date: 1/29/22

import pandas as pd
import ffmpeg
import numpy as np
from tqdm import tqdm
import av
import os

video_paths = ["/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/train/"]
# video_paths = ['/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/']

csv_paths = ["/BS/unintentional_actions/work/metadata/oops/splits/train_all_org.csv"]
# csv_paths = ['/BS/unintentional_actions/work/metadata/oops/splits/val_all_org.csv']

output_paths = [
    "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/train_downsize/"
]
# output_paths = ['/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val_downsize/']


def get_video_info(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    # video_fps = av.open(video_path).streams.video[0].average_rate
    return (
        int(video_stream["height"]),
        int(video_stream["width"]),
        int(int(video_stream["r_frame_rate"].split("/")[0]) / 100),
    )


def get_output_dim(h, w):
    size = 224  # resnet expects 122 and vit expects 224
    if h >= w:
        return int(h * size / w), size
    else:
        return size, int(w * size / h)


if __name__ == "__main__":

    for video_path, csv_path, output_path in zip(video_paths, csv_paths, output_paths):
        csv = pd.read_csv(csv_path)

        for index, row in tqdm(csv.iterrows(), total=csv.shape[0]):
            vid_path = video_path + row["filename"] + ".mp4"
            video_out_path = output_path + row["filename"] + ".mp4"
            if os.path.isfile(video_out_path):
                continue

            try:
                org_height, org_width, org_fps = get_video_info(vid_path)
                new_height, new_width = get_output_dim(org_height, org_width)

                if new_width % 2 != 0:
                    new_width -= 1
                if new_height % 2 != 0:
                    new_height -= 1

                ffmpeg.input(vid_path).filter("scale", new_height, new_width).output(
                    video_out_path
                ).global_args("-loglevel", "error").run()
            except Exception:
                print(vid_path)
