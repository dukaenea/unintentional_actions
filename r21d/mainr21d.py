# @Author: Enea Duka
# @Date: 11/12/21
import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')

from r21d.r21d_feat_loader import OopsR21DFeatDatast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.util_functions import Meter
import os
from utils.util_functions import Precision
from utils.model_saver import ModelSaver
from utils.arg_parse import opt
from utils.logging_setup import logger
import copy
from tqdm import tqdm
from utils.logging_setup import setup_logger_path
from datetime import datetime


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.head(x)


def test(model_, test_loader, loss_fn):
    model_.eval()
    prec = Precision('val')
    meter = Meter(mode='val', name='loss')

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):
            features = data['features'].squeeze()
            labels = data['labels'].squeeze()
            out = model_(features.cuda())
            loss = loss_fn(out, labels.cuda())

            meter.update(loss.item(), features.shape[0])
            prec.update_probs_sfx(out, labels.cuda())

    logger.debug('Val Loss: %f' % meter.avg)
    meter.reset()

    logger.debug('Val Acc: %f' % prec.top1(report_pca=False))
    return {'top1': prec.top1()}


def train(model_, loss_fn, optimizer_, train_loader_, test_loader):
    loss_meter = Meter('train')
    model_saver = ModelSaver(
        path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))

    logger.debug('Starting training for %d epochs:' % opt.epochs)

    # test(model_, test_loader, loss_fn)

    for epoch in range(opt.epochs):
        model_.train()

        for idx, data in enumerate(tqdm(train_loader_)):
            optimizer.zero_grad()
            features = data['features'].squeeze()
            labels = data['labels'].squeeze()
            out = model_(features.cuda())
            try:
                loss = loss_fn(out, labels.cuda())
            except Exception as e:
                print(features.shape)
                print(out.shape)

            loss_meter.update(loss.item(), features.shape[0])
            loss.backward()
            optimizer_.step()
        logger.debug('Train Loss: %f' % loss_meter.avg)
        loss_meter.reset()
        check_val = None

        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(model_, test_loader, loss_fn)

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'state_dict': copy.deepcopy(model_.state_dict()),
                             'optimizer': copy.deepcopy(optimizer_.state_dict().copy())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()


if __name__ == '__main__':
    opt.dataset = 'oops_r21d'
    opt.model_name = 'R21D_MLP'
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 128
    opt.sfx = str('%s.unint_act' % (opt.dataset))
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-8
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.01
    opt.weight_decay = 1e-8
    opt.test_freq = 1
    opt.save_model = 1
    opt.pretrained = True
    opt.log_name = 'lr:%f~ep:%d~bs:%d~ptr:%s_cntr_loss' % (opt.lr, opt.epochs, opt.batch_size, str(opt.pretrained))
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)
    opt.sfx = str('%s.unint_act.time%s_cntr_loss' % (opt.dataset, datetime.now().strftime('%Y%m%d-%H%M%S')))

    setup_logger_path()

    model = MLP()
    model.cuda()
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-8)
    loss = torch.nn.CrossEntropyLoss()

    train_set = OopsR21DFeatDatast('train')
    val_set = OopsR21DFeatDatast('val')
    train_loader = DataLoader(train_set,
                              batch_size=256,
                              num_workers=32,
                              shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=256,
                            num_workers=32,
                            shuffle=False)

    train(model, loss, optimizer, train_loader, val_loader)
