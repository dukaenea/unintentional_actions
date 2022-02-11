# @Author: Enea Duka
# @Date: 5/18/21

import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import os
import warnings
import torch
import copy

from utils.arg_parse import opt
from dataloaders.oops_loader import get_video_loader_frames
from datetime import datetime
from models.mlp import create_mlp_model
from models.r2plus1d import build_r2plus1d
from utils.logging_setup import setup_logger_path
from utils.util_functions import Meter, Precision
from utils.logging_setup import logger
from utils.model_saver import ModelSaver
from tqdm import tqdm
from utils.plotting_utils import visdom_plot_losses
from utils.logging_setup import viz


def test(**kwargs):
    model = kwargs['model']
    backbone = kwargs['backbone']
    loss = kwargs['loss']
    test_dataloader = kwargs['dataloader']
    mode = kwargs['mode']
    epoch = kwargs['epoch']
    time_id = kwargs['time']

    model.eval()
    prec = Precision(mode)
    meter = Meter(mode=mode, name='loss')

    with torch.no_grad():

        # if opt.use_tqdm:
        #     iterator = enumerate(tqdm(dataloader))
        # else:
        #     iterator = enumerate(dataloader)

        for idx, data in enumerate(tqdm(test_dataloader)):
            videos = data[0]
            labels = data[1]
            # labels[labels == -1] = 0

            back_out = backbone(videos.cuda())
            out = model(back_out)

            # if torch.any(torch.isnan(out)):
            #     print("found nan")
            #     continue

            _loss = loss(out, labels.cuda())
            meter.update(_loss.item(), videos.shape[0])
            prec.update_probs_sfx(out, labels.cuda())
            # if idx % 50 == 0:
            #     logger.debug("Iteration: %d" % idx)
            # if idx % 100 == 0:
            #     meter.log()
        meter.log()
        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
                               xylabel=('epoch', 'loss'), **meter.viz_dict())
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
                               xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper():  prec.top1()})

    return {'top1': prec.top1()}


def train(**kwargs):
    model = kwargs['model']
    backbone = kwargs['backbone']
    optimizer = kwargs['optimizer']
    loss_fn = kwargs['loss_fn']
    dataloader = kwargs['train_loader']
    tst = datetime.now().strftime('%Y%m%d-%H%M%S')

    loss_meter = Meter(mode='train')
    model_saver = ModelSaver(
        path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))
    logger.debug('Starting training for %d epochs:' % opt.epochs)
    backbone.eval()

    val_acc = test(model=model,
                   backbone=backbone,
                   loss=loss_fn,
                   dataloader=kwargs['val_dataloader'],
                   mode='val',
                   epoch=-1,
                   time=tst)

    # logger.debug("VAL ACC: %f" % val_acc['top1'])
    label_m1 = 0
    label_1 = 0
    label_2 = 0
    label_3 = 0
    for epoch in range(opt.epochs):
        model.train()
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)

        for idx, data in iterator:
            videos = data[0]
            labels = data[1]
            # labels[labels == -1] = 0
        #     counts = torch.bincount(labels)
        #     label_1 += counts[0]
        #     label_2 += counts[1]
        #     label_3 += counts[2]
        #
        # print(label_1)
        # print(label_2)
        # print(label_3)

            # labels[labels == -1] = 0
            optimizer.zero_grad()

            back_out = backbone(videos.cuda())
            out = model(back_out)

            loss = loss_fn(out, labels.cuda())
            loss_meter.update(loss.item(), labels.shape[0])
            loss.backward()
            optimizer.step()
        if epoch % opt.test_freq == 0 and epoch > 0:
            check_val = test(model=model,
                           backbone=backbone,
                           loss=loss_fn,
                           dataloader=kwargs['val_dataloader'],
                           mode='val',
                           epoch=-1,
                           time=tst)

            logger.debug("VAL ACC: %f" % val_acc['top1'])
        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()

        loss_meter.log()



def do_unint_act_task():

    opt.unint_act_backbone = 'r2plus1d'
    opt.dropout = 0.2
    opt.batch_size = 64
    opt.workers = 32
    opt.epochs = 20
    opt.balance_fails_only = True
    opt.all_fail_videos = False
    opt.viz = True
    opt.sfx = str('%s.unint_act_baseline.bs%d.epochs%d.%s' % ('oops', opt.batch_size, opt.epochs, datetime.now().strftime('%Y%m%d-%H%M%S')))

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-3
    opt.weight_decay = 1e-4
    opt.test_freq = 1


    opt.use_tqdm = True
    setup_logger_path()

    train_dataloader = get_video_loader_frames(opt)
    opt.val = True
    opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
    val_dataloader = get_video_loader_frames(opt)

    model, optimizer, loss = create_mlp_model(2048, 3)
    backbone, _, _ = build_r2plus1d()

    train(model=model,
          backbone=backbone,
          optimizer=optimizer,
          loss_fn=loss,
          train_loader=train_dataloader,
          val_dataloader=val_dataloader)




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    do_unint_act_task()













