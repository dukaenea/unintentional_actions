# @Author: Enea Duka
# @Date: 4/17/21
import sys
from dataloaders.dl_utils import tensor_to_zero_one
sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
from kinetics_loader import KineticsDataset
from rareacts_loader import RareactsDataset
from torch.utils.data import DataLoader
from pedestrian_loader import PedestrianDataset
from avenue_loader import AvenueDataset, test_dataset
from utils.util_functions import compute_mean_and_std
import torch
from oops_loader import VideoDataset

def test_dataloader():
    # val_dataset = RareactsDataset('val', 25, True, True)
    # val_dataloader = DataLoader(val_dataset,
    #                             num_workers=24,
    #                             batch_size=1,
    #                             shuffle=False)
    #
    # train_dataset = RareactsDataset('train', 25, True, True)
    # train_dataloader = DataLoader(train_dataset,
    #                               num_workers=24,
    #                               batch_size=1,
    #                               shuffle=False)
    #
    # test_dataset = RareactsDataset('test', 25, True, True)
    # test_dataloader = DataLoader(test_dataset,
    #                              num_workers=24,
    #                              batch_size=1,
    #                              shuffle=False)
    # train_stats = compute_mean_and_std(train_dataloader)
    # val_stats = compute_mean_and_std(val_dataloader)
    # test_stats = compute_mean_and_std(test_dataloader)
    #
    # all_stats = train_stats + val_stats + test_stats
    # mean = all_stats[0] / all_stats[2]
    # std = torch.sqrt((all_stats[1] / all_stats[2]) - mean.pow(2))
    #
    # print(mean, std)

    val_stats = compute_mean_and_std(val_dataloader)
    for idx, data in enumerate(tqdm(dataloader)):
        print(data['features'].shape)
        break

    ped_data = PedestrianDataset('Test')
    ped_dataloader = DataLoader(ped_data,
                                 num_workers=0,
                                 batch_size=1,
                                 shuffle=False)

    for idx, vid in enumerate(ped_dataloader):
        print(vid['video'].shape)
    # ave_data = train_dataset((256, 256), '/BS/unintentional_actions/nobackup/avenue/avenue/training')
    # ave_dataloader = DataLoader(ave_data,
    #                              num_workers=0,
    #                              batch_size=1,
    #                              shuffle=False)
    # for idx, vid in enumerate(ave_dataloader):
    #     a = vid

    # t = torch.randint(4, 220, (2, 2, 2, 2), dtype=torch.float64)
    # print(t)
    # t = tensor_to_zero_one(t)
    # print(t)


if __name__ == '__main__':
    test_dataloader()
