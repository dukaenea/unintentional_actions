# @Author: Enea Duka
# @Date: 6/25/21
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.arg_parse import opt

def calculate_memory_weights(model, dataset, loss):
    model.eval()
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=dataset.speed_and_motion_collate_fn)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            videos = data['features']
            labels = data['label']
            pure_nr_frames = data['pure_nr_frames']
            dsets = data['dset']

            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)
            out = model(videos, position_ids, None, pure_nr_frames)

            outs = torch.stack(list(torch.split(out, 4)))
            labels = torch.stack(list(torch.split(labels, 4)))
            dsets = torch.stack(list(torch.split(dsets, 4)))

            for idx, out in enumerate(outs):
                _loss = loss(out, labels[idx].cuda())
                model.module.update_memory_weights(_loss, dsets[idx])

    model.module.sharpen_normalize_memory_weights()


def calculate_memory(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=dataset.speed_and_motion_collate_fn)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            videos = data['features']
            pure_nr_frames = data['pure_nr_frames']
            dsets = data['dset']

            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)
            outs = model(videos, position_ids, None, pure_nr_frames)

            for idx, out in enumerate(outs):
                model.module.update_memory(out, dsets[idx])
