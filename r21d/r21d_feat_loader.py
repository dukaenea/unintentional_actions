# @Author: Enea Duka
# @Date: 11/12/21


import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np


class OopsR21DFeatDatast(Dataset):
    def __init__(self, mode):
        super(OopsR21DFeatDatast, self).__init__()

        self.feat_path = os.path.join('/BS/unintentional_actions/work/data/oops/r21d_features', mode)
        self.feat_list = sorted(os.listdir(self.feat_path))


    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat_path = self.feat_list[idx]
        npy_file = np.load(os.path.join(self.feat_path, feat_path), allow_pickle=True)
        out = {'features': torch.from_numpy(npy_file.item().get('feature')), 'labels': npy_file.item().get('label')}
        return out