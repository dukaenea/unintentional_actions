# @Author: Enea Duka
# @Date: 5/7/21

import torch

from utils.util_functions import Precision, label_idx_to_one_hot
from tqdm import tqdm
from utils.logging_setup import viz
from utils.plotting_utils import visdom_plot_losses
from utils.util_functions import Meter, contrastive_loss
from utils.arg_parse import opt
from utils.logging_setup import logger
from utils.util_functions import DistributionPlotter
from dataloaders.ucf_crime_loader import get_crime_video_loader
from transformer2x.trn_utils import prep_for_local, prep_for_global


def test(**kwargs):
    model = kwargs['model']
    loss = kwargs['loss']
    dataloader = kwargs['dataloader']
    mode = kwargs['mode']
    epoch = kwargs['epoch']
    time_id = kwargs['time']

    model.eval()
    prec = Precision(mode)
    meter = Meter(mode=mode, name='loss')
    bce_meter = Meter(mode=mode, name='bce_loss')


    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        for idx, data in iterator:
            keys = list(data.keys())
            videos = data.get(keys[0])
            # videos = videos.unsqueeze(1)
            labels = data['label']
            pure_nr_frames = data['pure_nr_frames']

            if opt.multi_scale:
                if labels.shape[0] == 1:
                    labels = labels.squeeze()
                else:
                    labels = labels.flatten()
                mask = (labels != -1)
                labels = labels[mask]
                labels = labels.type(torch.long)

                videos, position_ids, pnf = prep_for_local(videos, pure_nr_frames)
                if opt.backbone == 'vit_longformer':
                    videos, _ = model(videos, position_ids, None, pnf, labels, local=True, multi_scale=opt.multi_scale)
                else:
                    videos = model(videos, position_ids, None, pnf, labels, local=True, multi_scale=opt.multi_scale)
                if len(pure_nr_frames.shape) == 2:
                    pure_nr_frames = torch.t(pure_nr_frames)[0]
                videos, position_ids, pure_nr_frames, num_clips = prep_for_global(videos, position_ids,
                                                                                  pure_nr_frames)
                out = model(videos, position_ids, None, pure_nr_frames, labels, num_clips, False, video_level_pred=True)
            else:
                if opt.backbone == 'vit_longformer':
                    position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                        .expand(1, videos.shape[1]) \
                        .repeat(videos.shape[0], 1)
                    out = model(videos, position_ids, None, pure_nr_frames)
                    out = out.squeeze()
                elif opt.backbone == 'r3d_18':
                    out = model(videos.permute(0, 2, 1, 3, 4))

            _loss = loss(out, labels.cuda())
            meter.update(_loss.item(), videos.shape[0])

            if opt.task == 'classification':
                prec.update_probs_sfx(out, labels.cuda())
            else:
                prec.update_probs_reg_rel(out, labels.cuda())

        meter.log()
        bce_meter.log()

        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
                               xylabel=('epoch', 'loss'), **meter.viz_dict())
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
                               xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper():  prec.top1()})



    if opt.speed_and_motion:
        logger.debug('Epoch %d ==== Speed&Motoion Acc: %f' % (epoch, prec.top1()))
        return {'top1': prec.top1()}
    else:
        logger.debug('Val Acc: %f' % prec.top1())




    #
    # opt.dataset_path = '/BS/unintentional_actions/nobackup/ucf_crime'
    # opt.val = True
    # opt.frames_per_clip = 16
    # opt.step_between_clips_sec = 0.25
    # opt.fps_list = [30]
    # opt.workers = 32
    # opt.batch_size = 64
    # video_loader = get_crime_video_loader(opt)
    #
    # prec = Precision(mode)
    # bce_prec = Precision(mode)
    # meter = Meter(mode=mode, name='loss')
    # bce_meter = Meter(mode=mode, name='bce_loss')
    # con_loss_meter = Meter(mode=mode, name='loss')
    # if opt.speed_and_motion:
    #     prec2 = Precision(mode)
    #
    # if opt.task == 'regression':
    #     out_dist_plotter = DistributionPlotter()
    #
    # bce_loss = torch.nn.BCEWithLogitsLoss()
    #
    # all_labels = []
    # all_outs = []
    #
    # with torch.no_grad():
    #     if opt.use_tqdm:
    #         iterator = enumerate(tqdm(video_loader))
    #     else:
    #         iterator = enumerate(video_loader)
    #     for idx, data in iterator:
    #         keys = list(data.keys())
    #         videos = data.get(keys[0])
    #         # videos = videos.unsqueeze(1)
    #         labels = data['label']
    #         pure_nr_frames = data['pure_nr_frames']
    #         if opt.contrastive_loss:
    #             dsets = data['dset']
    #         # shuffle_idx = torch.randperm(videos.shape[0])
    #         # videos = videos[shuffle_idx]
    #         # pure_nr_frames = pure_nr_frames[shuffle_idx]
    #         # labels = labels[shuffle_idx]
    #
    #         # videos = videos.mean(4).mean(3)
    #         if opt.rep_backbone == 'resnet' and opt.rep_data_level != 'features':
    #             video_feats = []
    #             for video in videos:
    #                 video = video.permute(1, 0, 2, 3)
    #                 video_feats.append(feat_extractor(video))
    #             videos = torch.stack(video_feats)
    #             videos = videos.permute(0, 2, 1)
    #         position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
    #             .expand(1, videos.shape[1]) \
    #             .repeat(videos.shape[0], 1)
    #         _, out_bc = model(videos, position_ids, None, pure_nr_frames)
    #
    #         # if opt.task == 'regression':
    #         #     out = out.permute(1, 0).squeeze()
    #         #     out_dist_plotter.update_bins(out)
    #
    #         # _loss = loss(out, labels.cuda())
    #         # meter.update(_loss.item(), videos.shape[0])
    #
    #         bc_labels = labels.clone()
    #         bc_labels[bc_labels != 0] = 1
    #         bc_labels = bc_labels.unsqueeze(-1)
    #         bc_labels = bc_labels.float()
    #         _loss_bce = bce_loss(out_bc, bc_labels.cuda())
    #         bce_meter.update(_loss_bce.item(), videos.shape[0])
    #
    #         all_labels.append(bc_labels)
    #         all_outs.append(out_bc.cpu())
    #
    #         # if opt.contrastive_loss:
    #         #     c_loss = contrastive_loss(out, dsets)
    #         #     con_loss_meter.update(c_loss.item(), videos.shape[0])
    #         #     _loss += c_loss
    #         # if opt.task == 'classification':
    #         #     prec.update_probs_sfx(out, labels.cuda())
    #         # else:
    #         #     prec.update_probs_reg_rel(out, labels.cuda())
    #         bce_prec.update_probs_bc(out_bc, bc_labels.cuda())
    #         # if opt.debug and idx==10:
    #         #     break
    #         # if opt.speed_and_motion:
    #         #     break
    #         if idx == 200:
    #             break
    #     # meter.log()
    #     bce_meter.log()
    #     # if opt.task == 'regression':
    #     #     out_dist_plotter.plot_out_dist()
    #     # con_loss_meter.log()
    #     if opt.viz and epoch % opt.viz_freq == 0:
    #         visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
    #                            xylabel=('epoch', 'loss'), **meter.viz_dict())
    #         visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
    #                            xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper(): prec.top1()})
    #
    #     auc = prec.calculate_auc(torch.cat(all_outs, dim=0).squeeze(), torch.cat(all_labels, dim=0).squeeze())
    #     logger.debug('Epoch %d ==== AUC: %f    %f' % (epoch, auc[0], auc[1]))
    #
    # if opt.speed_and_motion:
    #     # logger.debug('Epoch %d ==== Speed&Motoion Acc: %f' % (epoch, prec.top1()))
    #     logger.debug('Epoch %d ==== BCE Acc: %f' % (epoch, bce_prec.top1()))
    #     # logger.debug('Val Acc Speed: %f' % prec2.top1())
    #     return {'top1': max(auc)}
    # # else:
    # #     logger.debug('Val Acc: %f' % prec.top1())
    # #     return {'top1': prec.top1()}
