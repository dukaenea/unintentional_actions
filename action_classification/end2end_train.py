
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
from end2end_test import test

def train(**kwargs):
    model = kwargs['model']
    train_loader = kwargs['train_loader']
    val_loader = kwargs['val_loader']
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    tst = datetime.now().strftime('%Y%m%d-%H%M%S')
    start_epoch = kwargs['epoch']


    loss_meter = Meter('train')
    model_saver = ModelSaver(path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))

    test(model=model,
         loss=loss,
         dataloader=val_loader,
         mode='val',
         time=tst,
         epoch=-1)
    logger.debug('Starting training for %d epochs:' % opt.epochs)

    # used to trigger Automatic Mixed Precision (amp) to speed up the training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        if opt.use_tqdm:
            iterator = enumerate(tqdm(train_loader))
        else:
            iterator = enumerate(train_loader)

        for idx, data in iterator:
            optimizer.zero_grad()
            videos = data['features']
            pure_nr_frames = data['pure_nr_frames']
            labels = data['label']

            # casts the operations in amp mode
            # with torch.cuda.amp.autocast():
                # videos = extract_features(videos, feat_extractor)

            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)
            # start_time = time.time()
            out = model(videos, position_ids, None, pure_nr_frames)

            _loss = loss(out, labels.cuda())
            loss_meter.update(_loss.item(), videos.shape[0])

            # Scales the loss, and calls backward()
            # to create scaled gradients
            # scaler.scale(_loss).backward()
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # Unscales gradients and calls
            # or skips optimizer.step()
            # scaler.step(optimizer)
            # Updates the scaler for the next iteration
            # scaler.update()
            optimizer.step()
            if idx % 100 == 0 and idx > 0:
                logger.debug("Train Loss: %f" % loss_meter.avg)
                # if idx == 500:
                #     break
            # total_time = time.time() - start_time
            # logger.debug("Time to run through model: %f" % total_time)
        logger.debug('Train Loss: %f' % loss_meter.avg)
        loss_meter.reset()
        # logger.debug('Train Acc: %f' % prec.top1())
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(model=model,
                             loss=loss,
                             dataloader=val_loader,
                             mode='val',
                             time=tst,
                             epoch=epoch)
            # logger.debug(str(loss.weight))
        if not opt.use_tqdm:
            print("=====================================================")

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'vtn_state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()

def extract_features(clips, feat_extractor):
    # start = time.time()
    c, ch, f, w, h = clips.shape
    # rearrange the batch
    clips = clips.permute(0, 2, 1, 3, 4)
    # concat all the clips of the videos
    clips = clips.reshape(c*f, ch, w, h)
    clips = feat_extractor(clips.cuda())

    clips = torch.stack(list(clips.split(16)))
    # total_time = time.time() - start
    # logger.debug("Time to extract the features: %f" % total_time)
    return clips
