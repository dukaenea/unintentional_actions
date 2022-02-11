
# @Author: Enea Duka
# @Date: 10/11/21

import torch

from tqdm import tqdm
from utils.logging_setup import viz
from utils.plotting_utils import visdom_plot_losses
from utils.util_functions import Meter, Precision
from utils.arg_parse import opt
from utils.logging_setup import logger
import matplotlib.pyplot as plt



def test(**kwargs):
    model = kwargs['model']
    loss = kwargs['loss']
    dataloader = kwargs['dataloader']
    mode = kwargs['mode']
    epoch = kwargs['epoch']
    time_id = kwargs['time']

    model.eval()
    prec = Precision(mode)
    bce_prec = Precision(mode)
    meter = Meter(mode=mode, name='loss')
    bce_meter = Meter(mode=mode, name='bce_loss')
    con_loss_meter = Meter(mode=mode, name='loss')
    if opt.speed_and_motion:
        prec2 = Precision(mode)


    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1/8]).cuda())
    all_labels = None
    all_outs = None


    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        for idx, data in iterator:
            keys = list(data.keys())
            videos = data.get(keys[0]).squeeze()
            # videos = videos.unsqueeze(1)
            boundries = data['boundries'].squeeze()
            pure_nr_frames = data['pure_nr_frames'].squeeze()
            vid_len = data['vid_len'].squeeze()

            # videos = videos.mean(4).mean(3)
            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)
            out, out_bc = model(videos, position_ids, None, pure_nr_frames, high_order_temporal_features=True)

            y_true = torch.zeros((vid_len,))
            y_pred = torch.zeros((vid_len,))

            segments_len = pure_nr_frames[0]
            for boundrie in boundries:
                if boundrie[0] != -1:
                    y_true[boundrie[0]: boundrie[1]] = 1

            for i in range(videos.shape[0]-1):
                seg_start = i * segments_len
                seg_end = (i+1) * segments_len
                i_scores = interpolate_scores(out_bc, i, segments_len)
                try:
                    y_pred[seg_start: seg_end] = out_bc[i]
                except Exception as e:
                    try:
                        y_pred[seg_start: vid_len] = out_bc[i]
                    except Exception as ex:
                        pass
            y_pred = smooth_scores(y_pred, segments_len//2)

            if all_labels is None:
                all_labels = y_true
                all_outs = y_pred
            else:
                all_labels = torch.cat([all_labels, y_true], dim=0)
                all_outs = torch.cat([all_outs, y_pred], dim=0)


            # if opt.task == 'regression':
            #     out = out.permute(1, 0).squeeze()
            #     out_dist_plotter.update_bins(out)

            # _loss = loss(out, labels.cuda())
            # meter.update(_loss.item(), videos.shape[0])

            # bc_labels = labels.clone()
            # bc_labels[bc_labels != 0] = 1
            # bc_labels = bc_labels.unsqueeze(-1)
            # bc_labels = bc_labels.float()
            # _loss_bce = bce_loss(out_bc, bc_labels.cuda())
            # bce_meter.update(_loss_bce.item(), videos.shape[0])

            # all_labels.append(bc_labels)
            # all_outs.append(out_bc.cpu())

            # if opt.task == 'classification':
            #     prec.update_probs_sfx(out, labels.cuda())
            # else:
            #     prec.update_probs_reg_rel(out, labels.cuda())
            # bce_prec.update_probs_bc(out_bc, bc_labels.cuda())
            # if opt.debug and idx==10:
            #     break
            # if opt.speed_and_motion:
            #     break
            # if idx == 200:
            #     break
        meter.log()
        bce_meter.log()
        # if opt.task == 'regression':
        #     out_dist_plotter.plot_out_dist()
        # con_loss_meter.log()
        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
                               xylabel=('epoch', 'loss'), **meter.viz_dict())
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
                               xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper():  prec.top1()})

        auc, fpr, tpr = prec.calculate_aucroc(all_outs.squeeze(), all_labels.squeeze())
        logger.debug('Epoch %d ==== AUC: %f' % (epoch, auc))

        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.4f)' % auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()

    # if opt.speed_and_motion:
    # logger.debug('Epoch %d ==== Speed&Motoion Acc: %f' % (epoch, prec.top1()))
    # logger.debug('Epoch %d ==== BCE Acc: %f' % (epoch, bce_prec.top1()))
    # logger.debug('Val Acc Speed: %f' % prec2.top1())
    return {'top1': auc}
    # else:
        # logger.debug('Val Acc: %f' % prec.top1())
        # return {'top1': prec.top1()}


def interpolate_scores(scores, i, segment_len):

    def _inter_couple(main_sc, secondary, perc, side):
        num_elms = int(main_sc.shape[0] * perc)
        if num_elms == 0:
            return main_sc

        main_elms = main_sc[:num_elms] if side == 'left' else main_sc[-num_elms:]
        sec_elms = secondary[-num_elms:] if side == 'left' else main_sc[:num_elms]

        weights = torch.arange(0, 1, 1/num_elms)
        if weights.shape[0] > main_elms.shape[0]:
            weights = weights[:main_elms.shape[0]]
        if side == 'left':
            weights = torch.flip(weights, dims=(0,))

        main_elms_w = main_elms * weights
        sec_elms_w = sec_elms * torch.flip(weights, dims=(0,))
        changed_elms = main_elms_w + sec_elms_w

        if side == 'left':
            main_sc[:num_elms] = changed_elms
        else:
            main_sc[-num_elms:] = changed_elms

        return main_sc



    center_score = scores[i]
    center_scores = [center_score.item()] * segment_len
    center_scores = torch.tensor(center_scores)
    if i > 0:
        left_score = scores[i-1]
        left_scores = [left_score.item()] * segment_len
        left_scores = torch.tensor(left_scores)
        center_scores = _inter_couple(center_scores, left_scores, 0.2, 'left')

    if i < scores.shape[0] - 1:
        right_score = scores[i + 1]
        right_scores = [right_score.item()] * segment_len
        right_scores = torch.tensor(right_scores)
        center_scores = _inter_couple(center_scores, right_scores, 0.2, 'right')

    return center_scores

def smooth_scores(scores, k):
    new_scores = []
    for i in range(scores.shape[0]):
        if i < k:
            new_scores.append(scores[i])
        elif i > (scores.shape[0] - k):
            new_scores.append(scores[i])
        else:
            s = (scores[i-k:i+k].sum(dim=-1)) / (2*k)
            new_scores.append(s)
    return torch.tensor(new_scores)

