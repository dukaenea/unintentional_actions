# @Author: Enea Duka
# @Date: 4/16/21

import torch
from torch.utils.data import Dataset
import torchvision
import os.path as pth
import ffmpeg
import av
import numpy as np
from torch.utils.data.dataloader import default_collate
# from utils.logging_setup import logger



# normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
#                         std=[0.5, 0.5, 0.5])
# unnormalize = T.Unnormalize(mean=[0.5, 0.5, 0.5],
#                             std=[0.5, 0.5, 0.5])
# train_transform = torchvision.transforms.Compose([
#     T.ToFloatTensorInZeroOne(),
#     T.Resize((128 * 2, 171 * 2)),
#     T.RandomHorizontalFlip(),
#     T.RandomRotate(),
#     normalize,
#     T.RandomCrop((224, 224))
# ])
# test_transform = torchvision.transforms.Compose([
#     T.ToFloatTensorInZeroOne(),
#     T.Resize((128 * 2, 171 * 2)),
#     normalize,
#     T.CenterCrop((112 * 2, 112 * 2))
# ])



class Normalize(object):
    def __init__(self, mean, std):
        if len(mean) > 1:
            self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
            self.std = torch.FloatTensor(std).view(1, 3, 1, 1)
        else:
            self.mean = torch.FloatTensor(mean).view(1, 1, 1)
            self.std = torch.FloatTensor(std).view(1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)


def tensor_to_zero_one(tensor):
    return tensor / 255 #(tensor - tensor.min()) / (tensor.max() - tensor.min())

class VidLoaderDs(Dataset):
    def __init__(self, mode, csv, oops_video_path):
        self.csv = csv
        self.oops_video_path = oops_video_path
        self.videos = {}
        self.mode = mode
        self.pth_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/%s_pth/' % mode

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        filename = self.csv['filename'][idx]
        # if dset == 'kinetics':
        #     video = self._load_video_features(self.kin_filename_to_path[filename])
        # else:
        vid_path = pth.join(self.oops_video_path, filename) + '.mp4'
        vid_path_pth = self.pth_path + filename + '.pth'
        if not pth.isfile(vid_path):
            raise
        if pth.isfile(vid_path_pth):
            video = torch.load(vid_path_pth)
        else:
            video = self._load_video_frames(vid_path)
            torch.save(video, vid_path_pth)
            logger.debug("Saved video %s: " % filename)
        return torch.empty((1, ))

    def _get_video_dim(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return int(video_stream['height']), int(video_stream['width'])
        except Exception as e:
            print(e)
            print(video_path)
            # return None, None
            raise

    def dummy_collate_fn(self, batch):
        return default_collate(batch)

    def _get_raw_video(self, video_path):
        height, width = self._get_video_dim(video_path)
        if width is None and height is None:
            return None
        container = av.open(video_path)
        cmd = (ffmpeg.input(video_path))
        video_fps = av.open(video_path).streams.video[0].average_rate
        cmd = cmd.filter('fps', fps=video_fps)

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, width, height, 3])
        video = torch.from_numpy(video)

        return video

    def _load_video_frames(self, video_path):
        video = self._get_raw_video(video_path)
        # video = video.permute(1, 0, 2, 3)
        if video is None:
            return None
        # video = train_transform(video) if self.mode == 'train' else test_transform(video)
        return video

    def save_videos(self, cache_path):
        torch.save(self.videos, cache_path)

    def load_videos_pth(self, cache_path):
        self.videos = torch.load(cache_path)

    def get_video(self, filename):
        return self.videos[filename]
