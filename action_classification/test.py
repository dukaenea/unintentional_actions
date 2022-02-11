# @Author: Enea Duka
# @Date: 5/7/21

import torch

from utils.util_functions import Precision, label_idx_to_one_hot
from tqdm import tqdm
from utils.logging_setup import viz
from utils.plotting_utils import visdom_plot_losses
from utils.util_functions import Meter
from utils.arg_parse import opt
from utils.logging_setup import logger
from utils.util_functions import DistributionPlotter
from transformer2x.trn_utils import prep_for_local, prep_for_crf


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
    if opt.task == 'regression':
        out_dist_plotter = DistributionPlotter()
    preds = []
    lbls = []
    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        total = 0
        for idx, data in iterator:
            videos = data['features']
            pure_nr_frames = data['pure_nr_frames']
            labels = data['label']

            if opt.use_crf:
                videos, position_ids, pnf = prep_for_local(videos, pure_nr_frames)
                out = model(videos, position_ids, None, pure_nr_frames)
                pure_nr_frames = torch.t(pure_nr_frames)[0]
                videos = prep_for_crf(out, pure_nr_frames)
                out, _loss = model(videos, None, None, pure_nr_frames, labels=labels, for_crf=True)
                _loss = _loss.mean()
                # out = torch.cat(out, dim=0)
                labels = labels.flatten()
                mask = (labels != -1)
                labels = labels[mask]
                labels = labels.type(torch.long)
            else:
                if opt.backbone == 'vit_longformer':
                    position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                        .expand(1, videos.shape[1]) \
                        .repeat(videos.shape[0], 1)
                    # videos = videos.mean(1).squeeze()
                    out = model(videos, position_ids, None, pure_nr_frames, classifier_only=False)
                else:
                    out = model(videos)

                # next_frames = data['next_frame']
                # if labels.shape[0] == 1:
                #     labels = labels.squeeze()
                # else:
                #     labels = labels.flatten()
                # mask = (labels != -1)
                # labels = labels[mask]
                # labels = labels.type(torch.long)
                # if labels.shape[0] == 1:
                #     labels = labels.squeeze()
                # if opt.mmargin_loss:
                _loss = loss(out, labels.cuda())
            # for o in out.cpu().squeeze():
            #     preds.append(o)
            # for l in labels.type(torch.float32):
            #     lbls.append(l)
            # else:
            # _loss = mmargin_contrastive_loss(out_c, labels.cuda())
            # ce_loss = loss(out, labels.cuda())
            # _loss = 10 * _loss + ce_loss
                # _loss = loss(out, labels.cuda())
            meter.update(_loss.item(), videos.shape[0])
            # if not opt.mmargin_loss:
            if opt.use_crf:
                prec.update_probs_crf(out, labels.cuda())
            else:
                prec.update_probs_sfx(out, labels.cuda(), report_pca=True, num_classes=3)
            # if idx % 100 == 0:
            #     meter.log()
            #     logger.debug('VAL Acc: %f' % prec.top1())
            # logger.debug(str(torch.abs(out).sum(1).sum(0)))
            # if idx == 50:
            #     break
        print(total)
        meter.log()
        # auc, eer = prec.calculate_auc(preds, lbls)
        # logger.debug('AUC: %f' % auc)
        # logger.debug('EER: %f' % eer)
        # if opt.task == 'regression':
        #     out_dist_plotter.plot_out_dist()
        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
                               xylabel=('epoch', 'loss'), **meter.viz_dict())
            # if not opt.mmargin_loss:
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
                               xylabel=('epoch', 'prec@1'), **{
                    'pr@1/%s' % mode.upper():  prec.top1(),
                    '(0)pr@1/%s' % mode.upper(): prec.lab_class[0]['correct']/prec.lab_class[0]['total'],
                    '(1)pr@1/%s' % mode.upper(): prec.lab_class[1]['correct']/prec.lab_class[1]['total'],
                    '(2)pr@1/%s' % mode.upper(): prec.lab_class[2]['correct']/prec.lab_class[2]['total'],
                    })
    # if not opt.mmargin_loss:
    logger.debug('Val Acc: %f' % prec.top1(report_pca=True))
    # return {'top1': (1 / meter.avg)}
    return {'top1': prec.top1()}
