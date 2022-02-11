
# @Author: Enea Duka
# @Date: 9/1/21

import torch
import os
import copy
import time
from tqdm import tqdm
from datetime import datetime
from utils.util_functions import Meter
from utils.model_saver import ModelSaver
from utils.arg_parse import opt
from utils.logging_setup import logger
from rep_learning.end2end_test import test
from utils.util_functions import lr_func_cosine
import random


def train(**kwargs):
    model = kwargs['model']
    feat_extractor = kwargs['feat_extractor']
    train_loader = kwargs['train_loader']
    val_loader = kwargs['val_loader']
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    tst = datetime.now().strftime('%Y%m%d-%H%M%S')
    start_epoch = kwargs['epoch']
    train_set = kwargs['train_set']

    loss_meter = Meter('train')
    model_saver = ModelSaver(path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))
    cached_videos = None
    # best_check_val = test(model=model,
    #                      feat_extractor=feat_extractor,
    #                      loss=loss,
    #                      dataloader=val_loader,
    #                      mode='val',
    #                      time=tst,
    #                      epoch=-1)
    logger.debug('Starting training for %d epochs:' % opt.epochs)

    # used to trigger Automatic Mixed Precision (amp) to speed up the training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        # model.module.backbone.eval()
        # if len(train_set.val_idx.keys())>1:
        #     random.shuffle(train_set.val_idx)
        # new_lr = lr_func_cosine(opt.lr, opt.lr * opt.cos_decay_lr_factor, opt.epochs, epoch)
        # logger.debug("New LR for epoch %d: %f" % (epoch, new_lr))
        # for param in optimizer.param_groups:
        #     param['lr'] = new_lr

        if opt.use_tqdm:
            iterator = enumerate(tqdm(train_loader))
        else:
            iterator = enumerate(train_loader)

        for idx, data in iterator:
            optimizer.zero_grad()
            videos = data['features']
            if torch.isnan(videos).any():
                logger.debug('Found nan!!')
                continue
            pure_nr_frames = data['pure_nr_frames']
            labels = data['label']

            # casts the operations in amp mode
            with torch.cuda.amp.autocast():
                # videos = extract_features(videos, feat_extractor, pure_nr_frames)

                position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                    .expand(1, videos.shape[1]) \
                    .repeat(videos.shape[0], 1)
                # start_time = time.time()
                out = model(videos, position_ids, None, pure_nr_frames)

                _loss = loss(out, labels.cuda())
            loss_meter.update(_loss.item(), videos.shape[0])

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scaler for the next iteration
            scaler.update()
            if idx % 200 == 0 and idx > 0:
                logger.debug("Train Loss: %f" % loss_meter.avg)
                # break
            # total_time = time.time() - start_time
            # logger.debug("Time to run through model: %f" % total_time)
        logger.debug('Train Loss: %f' % loss_meter.avg)
        loss_meter.reset()
        # logger.debug('Train Acc: %f' % prec.top1())
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(model=model,
                                             feat_extractor=feat_extractor,
                                             loss=loss,
                                             dataloader=val_loader,
                                             mode='val',
                                             time=tst,
                                             epoch=epoch)
            # logger.debug(str(loss.weight))

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            # if check_val[] > best_check_val:
            #     save_dict = {'epoch': epoch,
            #                  'vtn_state_dict': model.state_dict(),
            #                  'vit_state_dict': feat_extractor.state_dict(),
            #                  'optimizer': optimizer.state_dict()}
            #     path = os.path.join(os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'), key,
            #                        '%s_%s_%s_v%.4f_ep%d.pth.tar' % (opt.model_name, opt.pfx, opt.sfx, check_val, epoch))
            #     torch.save(save_dict, )
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'vtn_state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()


def extract_features(clips, feat_extractor, clip_lens):
    # start = time.time()
    c, f, ch, w, h = clips.shape
    # rearrange the batch
    # clips = clips.permute(0, 2, 1, 3, 4)
    # concat all the clips of the videos
    clips = clips.reshape(c*f, ch, w, h)
    clips = feat_extractor(clips.cuda())

    split_size = torch.max(clip_lens)
    clips = torch.stack(list(clips.split(f)))
    # total_time = time.time() - start
    # logger.debug("Time to extract the features: %f" % total_time)
    return clips
