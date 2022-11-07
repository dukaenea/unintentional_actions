# @Author: Enea Duka
# @Date: 4/20/21

from torch.utils.data import Dataset
from utils.util_functions import Labels
from dataloaders.dl_utils import Normalize, tensor_to_zero_one
import os.path as ops
import numpy as np
import pandas as pd
import ffmpeg
import torch
import av


class RareactsDataset(Dataset):
    def __init__(
        self, mode, fps, spat_crop, hflip, norm_statistics, video_root_path=None
    ):
        super(RareactsDataset, self).__init__()

        self.mode = mode
        self.fps = fps
        self.spat_crop = spat_crop
        self.spat_scale = False
        self.hflip = hflip
        self.size = 224
        self.n_splits = 1
        self.training = True if mode == "train" else False
        self.norm = Normalize(mean=norm_statistics["mean"], std=norm_statistics["std"])

        self.classes_path = "../resources/data/rareacts/splits/train_classes.txt"
        self.csv_path = "../resources/data/rareacts/splits/%s.csv" % mode
        if video_root_path is None:
            raise ValueError("Invalid value for video root path.")

        self.video_path = "%s/%s" % (video_root_path, mode)

        self.labels = Labels(self.classes_path)
        self.csv = pd.read_csv(self.csv_path)
        self.nclasses = len(self.labels)

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        return int(video_stream["height"]), int(video_stream["width"])

    def _get_output_dim(self, h, w):
        if self.spat_scale and self.training:
            size = np.random.randint(self.size + 2, max(h, w, self.size) * 2)

        else:
            size = self.size
        if h >= w:
            return int(h * size / w), size
        else:
            return size, int(w * size / h)

    def _preprocess_video(self, tensor):
        tensor = tensor_to_zero_one(tensor)
        tensor = self.norm(tensor)
        return tensor

    def _get_video(self, video_path):
        h, w = self._get_video_dim(video_path)
        height, width = self._get_output_dim(h, w)
        cmd = ffmpeg.input(video_path).filter("scale", width, height)
        video_fps = av.open(video_path).streams.video[0].average_rate
        cmd = cmd.filter("fps", fps=video_fps)

        if self.spat_crop:
            if not self.training:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
                height, width = self.size, self.size
            else:
                if (width - self.size) // 2 <= 0:
                    x = 0
                else:
                    x = np.random.randint(0, (width - self.size) // 2)

                if (height - self.size) // 2 <= 0:
                    y = 0
                else:
                    y = np.random.randint(0, (height - self.size) // 2)

                cmd = cmd.crop(x, y, self.size, self.size)
                height, width = self.size, self.size

        if self.hflip:
            if np.random.rand() > 0.5:
                cmd = cmd.hflip()

        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = torch.from_numpy(video.astype("float32"))
        video = video.permute(0, 3, 1, 2)

        return video

    def __getitem__(self, idx):

        output = {}
        try:
            video_name = ops.splitext(self.csv.iloc[idx]["filename"])[0] + ".mp4"
            video_path = ops.join(self.video_path, video_name)

            output["features"] = self._get_video(video_path)
            output["label"] = self.labels[self.csv["label"][idx]]
            output["filename"] = self.csv["filename"][idx]
        except Exception as e:
            print(e)
            print(self.csv.iloc[idx]["filename"])
            return {"filename": self.csv.iloc[idx]["filename"]}
        return output
