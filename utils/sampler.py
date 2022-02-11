
# @Author: Enea Duka
# @Date: 5/18/21
import math
import random

import torch
import torch.distributed as dist
import torchvision.datasets.video_utils
from torch.utils.data import Sampler
import utils.my_video_utils


class DistributedSampler(Sampler):
    """
    Extension of DistributedSampler, as discussed in
    https://github.com/pytorch/pytorch/issues/23430
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if isinstance(self.dataset, Sampler):
            orig_indices = list(iter(self.dataset))
            indices = [orig_indices[i] for i in indices]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class UniformClipSampler(torch.utils.data.Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video, equally spaced
    Arguments:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """

    def __init__(self, video_clips, max_clips_per_video):
        if not isinstance(video_clips, torchvision.datasets.video_utils.VideoClips) and not isinstance(video_clips, utils.my_video_utils.VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, uniformly spaced
        for c in self.video_clips.clips:
            length = len(c)
            step = max(length // self.max_clips_per_video, 1)
            sampled = torch.arange(length)[::step] + s
            s += length
            idxs.append(sampled)
        idxs = torch.cat(idxs).tolist()
        return iter(idxs)

    def __len__(self):
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)


class RandomVideoSampler(torch.utils.data.Sampler):
    def __init__(self, video_clips):
        if not isinstance(video_clips, torchvision.datasets.video_utils.VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        # idxs = torch.arange(0, self.video_clips.num_videos())
        idxs = self._get_filtered_idxs()
        r = torch.randperm(idxs.shape[0])
        idxs = idxs[r]
        return iter(idxs)

    def __len__(self):
        return self._get_filtered_idxs().shape[0]

    def _get_filtered_idxs(self):
        all_idx = list(range(self.video_clips.num_videos()))
        filtered_idx = []
        for idx in all_idx:
            num_clips = len(self.video_clips.clips[idx])
            if num_clips > 0:
                filtered_idx.append(idx)
        return torch.LongTensor(filtered_idx)



class RandomClipSampler(torch.utils.data.Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly
    Arguments:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """

    def __init__(self, video_clips, max_clips_per_video):
        if not isinstance(video_clips, torchvision.datasets.video_utils.VideoClips) and not isinstance(video_clips, utils.my_video_utils.VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs))
        idxs = idxs[perm].tolist()
        return iter(idxs)

    def __len__(self):
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)
        # return 100



class ConcatSampler(torch.utils.data.Sampler):
    def __init__(self, samplers):
        # to be used with concatdataset, if you need different sampler for each dataset in the concat
        # order of samplers must be same as order of corresponding datasets in concatdataset
        assert type(samplers) is list and len(samplers) > 0, 'must pass in list of 1 or more samplers'
        self.samplers = samplers
        self.cum_lens = torch.LongTensor([len(s.video_clips) for s in samplers]).cumsum(0)

    def __iter__(self):
        idxs = list(iter(self.samplers[0]))
        for s, l in zip(self.samplers[1:], self.cum_lens):
            idxs.extend(map(lambda i: (l+i).item(), iter(s)))
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return sum(len(_) for _ in self.samplers)