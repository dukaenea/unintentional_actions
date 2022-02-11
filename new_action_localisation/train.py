
# @Author: Enea Duka
# @Date: 5/7/21

import torch
import os
import copy
from tqdm import tqdm
from datetime import datetime
from new_action_localisation.test import test
import torch.nn.functional as F
from utils.util_functions import label_idx_to_one_hot, adjust_lr, AIAYNScheduler
from utils.model_saver import ModelSaver
from utils.util_functions import Meter
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.util_functions import lr_func_cosine
from utils.util_functions import Precision
from transformer2x.trn_utils import prep_for_local, prep_for_crf
import math

from models.pm_vtn import freeze_model, unfreeze_model


def train(**kwargs):
    model = kwargs['model']
    train_loader = kwargs['train_loader']
    val_loader = kwargs['val_loader']
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    tst = datetime.now().strftime('%Y%m%d-%H%M%S')
    s_epoch = kwargs['epoch']

    loss_meter = Meter('train')
    loc_loss_meter = Meter('train')
    model_saver = ModelSaver(path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))
    # if opt.lr_scheduler == 'aiayn': # Attention is All You Need
    aiayn_scheduler = AIAYNScheduler(opt.hidden_dim,
                                     0.3 * (len(train_loader) * opt.epochs)) # 10% of the steps for the warmup
    # test(model=model,
    #      loss=loss,
    #      dataloader=val_loader,
    #      mode='val',
    #      time=tst,
    #      epoch=-1)
    # logger.debug(str(loss.weight))
    logger.debug('Starting training for %d epochs:' % opt.epochs)
    new_lr = opt.lr
    prec = Precision('train')

    for epoch in range(s_epoch, opt.epochs):
        model.train()

        new_lr = lr_func_cosine(opt.lr, opt.lr * opt.cos_decay_lr_factor, opt.epochs, epoch)
        logger.debug("New LR: %f" % new_lr)
        optimizer.param_groups[0]['lr'] = new_lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = new_lr * opt.backbone_lr_factor

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lr

        if opt.use_tqdm:
            iterator = enumerate(tqdm(train_loader))
        else:
            iterator = enumerate(train_loader)

        for idx, data in iterator:
            optimizer.zero_grad()
            videos = data['features']
            pure_nr_frames = data['pure_nr_frames']
            labels = data['label']
            labels_trn = data['wif']

            if opt.use_crf:
                videos, position_ids, pnf = prep_for_local(videos, pure_nr_frames)
                out = model(videos, position_ids, None, pure_nr_frames)
                pure_nr_frames = torch.t(pure_nr_frames)[0]
                videos = prep_for_crf(out, pure_nr_frames)
                _loss = model(videos, None, None, pure_nr_frames, labels=labels, for_crf=True).mean()
            else:
                position_ids = torch.tensor(list(range(0, videos.shape[1])))\
                                    .expand(1, videos.shape[1])\
                                    .repeat(videos.shape[0], 1)
                # videos = videos.mean(1).squeeze()
                out, features = model(videos, position_ids, None, pure_nr_frames, classifier_only=False)
                out_trn, labels_trn = filter_transitions(out, features, labels_trn)
                out_trn = model(out_trn)
                 # next_frames = data['next_frame']
                # try:
                _loss = loss(out_trn, labels_trn.cuda())
                loc_loss_meter.update(_loss.item(), out_trn.shape[0])
                _loss_cl = loss(out, labels.cuda())
                loss_meter.update(_loss_cl.item(), videos.shape[0])
                _loss += _loss_cl

                _loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                # if idx % 100 == 0:
                #     logger.debug("Train Loss: %f" % loss_meter.avg)
                # if idx == 100:
                #     break
        logger.debug('Train Loss: %f' % loss_meter.avg)
        loss_meter.reset()
        # logger.debug('Train Acc: %f' % prec.top1())
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(model=model,
                             loss=loss,
                             dataloader=val_loader,
                             time=tst,
                             mode='val',
                             epoch=epoch)
            # logger.debug(str(loss.weight))
        if not opt.use_tqdm:
            print("=====================================================")

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()


def filter_transitions(outputs, features, labels_trn):
    pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    trn_idx = pred_labels == 1
    return features[trn_idx], labels_trn[trn_idx]