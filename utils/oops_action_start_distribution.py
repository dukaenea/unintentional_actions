
# @Author: Enea Duka
# @Date: 6/14/21
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
import math
from utils.arg_parse import opt

from matplotlib import pyplot as plt

if __name__ == '__main__':
    opt.task='regression'
    train_set = SimpleOopsDataset('train', 25, 'frames', True,
                                  {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    val_set = SimpleOopsDataset('val', 25, 'frames', True,
                                {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

    train_dataloader = DataLoader(train_set,
                                  batch_size=16,
                                  num_workers=16,
                                  shuffle=False,
                                  collate_fn=train_set.pad_videos_collate_fn)
    val_dataloader = DataLoader(val_set,
                                  batch_size=16,
                                  num_workers=16,
                                  shuffle=False,
                                  collate_fn=train_set.pad_videos_collate_fn)

    train_bins = [0] * 11
    val_bins = [0] * 11

    train_ua = 0
    train_ia = 0

    val_ua = 0
    val_ia = 0

    for idx, data in tqdm(enumerate(train_dataloader)):
        rel_ts = data['rel_t']
        ts = data['t']
        for t in ts:
            if t == -1:
                train_ia += 1
            else:
                train_ua += 1
        for rel_t in rel_ts:
            try:
                idx = math.floor(rel_t * 10)
                if train_bins[idx] is None:
                    train_bins[idx] = 1
                else:
                    train_bins[idx] += 1
            except Exception:
                print('here')

    for idx, data in tqdm(enumerate(val_dataloader)):
        rel_ts = data['rel_t']
        ts = data['t']
        for t in ts:
            if t == -1:
                val_ia += 1
            else:
                val_ua += 1
        for rel_t in rel_ts:
            idx = math.floor(rel_t * 10)
            if val_bins[idx] is None:
                val_bins[idx] = 1
            else:
                val_bins[idx] += 1

    train_bins = [tb / sum(train_bins) for tb in train_bins]
    val_bins = [vb / sum(val_bins) for vb in val_bins]
    x = range(11)
    plt.bar(x, train_bins, color='red')
    plt.title('Trai split relative action start time distribution')
    plt.xlabel('Relative video time')
    plt.ylabel('% Videos/Bin')
    plt.show()

    plt.bar(x, val_bins, color='red')
    plt.title('Validation split relative action start time distribution')
    plt.xlabel('Relative video time')
    plt.ylabel('% Videos/Bin')
    plt.show()

    print(train_ua/(train_ia + train_ua))
    print(train_ia/(train_ia + train_ua))
    print(val_ua/(val_ia + val_ua))
    print(val_ia/(val_ia + val_ua))