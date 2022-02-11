
# @Author: Enea Duka
# @Date: 10/19/21
import torch
import os
import copy
from tqdm import tqdm
from datetime import datetime
from crime_detection_i3d.test import test
from utils.util_functions import AIAYNScheduler
from utils.model_saver import ModelSaver
from utils.util_functions import Meter
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.util_functions import lr_func_cosine
from utils.util_functions import Precision
from utils.util_functions import DistributionPlotter
import torch.nn as nn
import random

def train(**kwargs):

    model = kwargs['model']
    train_normal_loader = kwargs['train_normal_loader']
    train_abnormal_loader = kwargs['train_abnormal_loader']
    val_loader = kwargs['val_loader']
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    tst = datetime.now().strftime('%Y%m%d-%H%M%S')
    s_epoch = kwargs['epoch']

    loss_meter = Meter('train')
    bce_meter = Meter(mode='train', name='bce_loss')
    model_saver = ModelSaver(
        path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))

    test(model=model,
         dataloader=val_loader)

    logger.debug('Starting training for %d epochs:' % opt.epochs)
    prec = Precision('train')
    new_lr = opt.lr

    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1 / 8]).cuda())

    if opt.task == 'regression':
        out_dist_plotter = DistributionPlotter()

    lrs = [1e-4, 8e-5, 5e-5, 3e-5, 1e-5]
    curr_lr = opt.lr

    for epoch in range(s_epoch, opt.epochs):
        model.train()
        # if opt.lr_scheduler == 'step':
        #     if epoch % 10 == 0:
        #         new_lr = adjust_lr(optimizer, new_lr, 0.1)
        new_lr = lr_func_cosine(opt.lr, opt.lr * opt.cos_decay_lr_factor, opt.epochs, epoch)
        # if epoch == 2:
        #     curr_lr = 8e-5
        # elif epoch == 10:
        #     curr_lr = 5e-5
        # elif epoch == 20:
        #     curr_lr = 1e-5

        logger.debug("New LR: %f" % new_lr)
        optimizer.param_groups[0]['lr'] = new_lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = new_lr * opt.backbone_lr_factor

        # if opt.use_tqdm:
        #     iterator = enumerate(tqdm(train_loader))
        # else:
        #     iterator = enumerate(train_loader)

        iter_len = len(train_normal_loader)
        train_normal_iterator = iter(train_normal_loader)
        train_abnormal_iterator = iter(train_abnormal_loader)

        for _ in tqdm(range(iter_len)):

            normal_videos, normal_labels = next(train_normal_iterator)
            anomal_videos, anomal_labels = next(train_abnormal_iterator)


            videos = torch.cat((normal_videos, anomal_videos), dim=0)
            videos_shape = videos.shape

            videos = couple_batch_to_clip_batch(videos, videos_shape)
            pure_nr_frames = torch.tensor([32]*videos.shape[0]).to(videos.device)

            # videos = videos.unsqueeze(1)

            # a_videos = torch.stack(list(torch.split(videos, videos.shape[0]//4)))
            # a_pnf = torch.stack(list(torch.split(pure_nr_frames, videos.shape[0]//4)))
            # a_labels = torch.stack(list(torch.split(labels, videos.shape[0]//4)))

            # for i in range(a_videos.shape[0]):
            #     videos = a_videos[i]
            #     pure_nr_frames = a_pnf[i]
            #     labels = a_labels[i]

            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)

            # pure_nr_frames = torch.stack(list(pure_nr_frames.split(videos.shape[1])), dim=0)
            outs = []
            features = []
            # for idx in range(0, videos.shape[0], 4):
            #     video = videos[idx:idx+4]
            #     pid = position_ids[idx:idx+4]
            #     pnf = pure_nr_frames[idx:idx+4]
            #     out, out_bc, feats = model(video, pid, None, pnf)
            #     outs.extend(out_bc.split(1))
            #     features.extend(feats.split(1))
                # logger.debug("Extracting for idx %d" % idx)
            # logger.debug("Videos: %d ---- Outs: %d" % (videos.shape[0], len(outs)))
            out, out_bc = model(videos, position_ids, None, pure_nr_frames, high_order_temporal_features=False)


            # score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            # feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(videos)


            out = out.squeeze()
            # out_bc = torch.stack(outs, dim=0)
            # out_features = torch.stack(features, dim=0)
            # out_bc = clip_batch_to_couple_batch(out_bc, videos_shape)
            # out_bc, out_features = clip_batch_to_vdeo_batch(out_bc, out_features, videos_shape)
            # _loss = calc_loss(out_bc, out_features, labels, 3, 1e-4, 100)
            _loss = loss(out_bc, opt.batch_size)

            loss_meter.update(_loss.item(), videos.shape[0]//2)
            # bc_labels = labels.clone()
            # bc_labels[bc_labels != 0] = 1
            # bc_labels = bc_labels.unsqueeze(-1)
            # bc_labels = bc_labels.float()
            # _loss_bce = bce_loss(out_bc, bc_labels.cuda())
            # bce_meter.update(_loss_bce.item(), videos.shape[0])
            #
            # _loss += _loss_bce
            _loss.backward()
            # if (idx + 1) % 32 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            # prec.update_probs_reg_rel(out, labels.cuda())

            # if idx % 50 == 0 and idx > 0:
            #     logger.debug('Loss: %f' % loss_meter.avg)
            #     logger.debug('BCE Loss: %f' % bce_meter.avg)
            # if idx == 300:
            #     break
            # if opt.debug and idx == 10:
            #     break
            # if opt.speed_and_motion:
            #     break

        logger.debug('Loss: %f' % loss_meter.avg)
        loss_meter.reset()
        bce_meter.reset()
        check_val = None
        # logger.debug('Epoch %d ==== Speed&Motoion TrainAcc: %f' % (epoch, prec.top1()))
        if opt.task == 'regression':
            out_dist_plotter.plot_out_dist()
        if (opt.test_freq and epoch % opt.test_freq == 0) or epoch == (opt.epochs - 1):
            opt.rep_learning = False
            opt.viz = False
            check_val = test(model=model,
                            dataloader=val_loader)
            # opt.rep_learning = True
            # opt.viz = True
            # test_classify(model=model,
            #              feat_extractor=feat_extractor,
            #              loss=loss,
            #              dataloader=val_loader_class,
            #              mode='val',
            #              time=tst,
            #              epoch=epoch)

        if not opt.use_tqdm:
            print("=====================================================")

        # if ((opt.save_model and epoch % opt.save_model == 0) or epoch == (opt.epochs - 1)) and check_val:
        #     if model_saver.check(check_val):
        #         save_dict = {'epoch': epoch,
        #                      'state_dict': copy.deepcopy(model.state_dict()),
        #                      'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
        #         model_saver.update(check_val, save_dict, epoch)
        #
        #     model_saver.save()

    return model


def couple_batch_to_clip_batch(couple_batch, samples_shape):
    B, CR, F, C = samples_shape
    couple_batch = couple_batch.reshape(B * CR, F, C)
    return couple_batch


def clip_batch_to_couple_batch(clip_batch, samples_shape):
    B, CO, CL, F, C = samples_shape
    clip_batch = torch.stack(list(clip_batch.split(CL)), dim=0)
    clip_batch = torch.stack(list(clip_batch.split(CO)), dim=0)
    clip_batch = clip_batch.squeeze()
    return clip_batch

def clip_batch_to_vdeo_batch(clip_batch, features_batch, samples_shape):
    B, CO, CL, F, C = samples_shape
    clip_batch = torch.stack(list(clip_batch.split(CL)), dim=0)
    features_batch = torch.stack(list(features_batch.split(CL)), dim=0)
    return clip_batch, features_batch

def calc_loss(scores, features, bag_labels, k, alpha, margin):

    def _select_feats(feats, idxs):
        sel_feats = None
        for feat, idx in zip(feats, idxs):
            if sel_feats is None:
                sel_feats = feat[idx]
            else:
                sel_feats = torch.stack([sel_feats, feat[idx]], dim=0)

        return sel_feats

    feat_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()

    scores = scores.squeeze()
    features = features.squeeze()
    bag_labels = bag_labels.flatten()

    # first get the magnitude of the features
    magnitudes = torch.norm(features, p=2, dim=2) # features -> [B, SEGS, F]

    # first separate the normal from the abnormal bags
    nrm_scores = scores[bag_labels == 0]
    anm_scores = scores[bag_labels == 1]

    nrm_bags = features[bag_labels == 0]
    anm_bags = features[bag_labels == 1]

    nrm_magnitudes = magnitudes[bag_labels == 0]
    anm_magnitudes = magnitudes[bag_labels == 1]

    # get the top-k magnitudes for each bag
    nrm_idx = torch.topk(nrm_magnitudes, k, dim=1)[1]
    anm_idx = torch.topk(anm_magnitudes, k, dim=1)[1]

    # reshape the indexes to point to the features
    # nrm_feat_idx = nrm_idx.unsqueeze(2).expand(-1, -1, nrm_bags.shape[2])
    # anm_feat_idx = anm_idx.unsqueeze(2).expand(-1, -1, anm_bags.shape[2])

    # get the features with the largest magnitude
    nrm_features = _select_feats(nrm_bags, nrm_idx)
    anm_features = _select_feats(anm_bags, anm_idx)

    # get the scores for the largest magnitudes
    nrm_scores = _select_feats(nrm_scores, nrm_idx)
    anm_scores = _select_feats(anm_scores, anm_idx)

    sc_labels = torch.tensor([0] * nrm_scores.shape[0] * nrm_scores.shape[1] + [1] * anm_scores.shape[0] * anm_scores.shape[1], dtype=torch.float32).to(nrm_scores.device)
    all_scores = torch.cat((nrm_scores.flatten(), anm_scores.flatten()))
    scores_loss = bce_loss_fn(all_scores, sc_labels)

    loss_abn = torch.abs(margin - torch.norm(torch.mean(anm_features, dim=1), p=2, dim=1))
    loss_nrm = torch.norm(torch.mean(nrm_features, dim=1), p=2, dim=1)
    feat_loss = torch.mean((loss_abn + loss_nrm) ** 2)

    final_loss = scores_loss + alpha * feat_loss

    # sparsity constraint
    sp_loss = torch.mean(torch.norm(anm_scores, dim=0))

    # smoothness constraint
    smoothed_scores = anm_scores[:, 1:] - anm_scores[:, :-1]
    smoothed_scores = smoothed_scores.pow(2).sum(dim=-1).mean()

    return final_loss + 8e-3 * sp_loss + 8e-4 * smoothed_scores


