    # @Author: Enea Duka
# @Date: 6/30/21


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
from visualization.visualize_unint_action_localization import visualize_perdiction
import statistics
from visualization.tsne_visualizer import do_tsne


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

    outputs = {}

    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        total = 0
        zeros = 0
        ones = 1
        twos = 2
        all_outs = []
        all_labels = []
        for idx, data in iterator:
            videos = data['features'].squeeze()
            pure_nr_frames = data['pure_nr_frames'].squeeze()
            # video_names = data['video_name']
            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)
            out = model(videos, position_ids, None, pure_nr_frames, return_features=True)
            labels = data['label'].squeeze()
            all_outs.extend(list(out.detach().cpu()))
            all_labels.extend(labels)
            # # if idx == 100:
            # #     break
            # continue
            # trn_times = data['t']
            # clip_boundries = data['clip_time_boarders'].squeeze()
            _loss = loss(out, labels.cuda())

            # if idx % 200 == 0 and idx > 0:
            #     logger.debug('Val Acc: %f' % prec.top1())
            for o in out:
                o =torch.softmax(o, dim=0)
                total += 1
                am = torch.argmax(o)
                if am == 0:
                    zeros += 1
                elif am == 1:
                    ones += 1
                else:
                    twos += 1

            meter.update(_loss.item(), videos.shape[0])
            prec.update_probs_sfx(out, labels.cuda(), report_pca=True, num_classes=3)
            # prec.update_probs_loc_class(out, labels.cuda(), trn_times.cuda(), clip_boundries.cuda())

            outputs = keep_relevant_outs(out, data, outputs)
        all_outs = [o.numpy() for o in all_outs]
        all_labels = [l.item() for l in all_labels]
        all_labels = torch.tensor(all_labels)
        do_tsne(all_outs, all_labels, 'Feature Space (F2C) + TTIBUA')

        # print("Class predictions: %f, %f, %f" % (zeros/total, ones/total, twos/total))
        # print(total)
        # meter.log()
        # logger.debug('Val Acc: %f' % prec.top1(report_pca=True))
        # calc_acc(outputs)


        # if opt.task == 'regression':
        #     out_dist_plotter.plot_out_dist()
    #     if opt.viz and epoch % opt.viz_freq == 0:
    #         visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
    #                            xylabel=('epoch', 'loss'), **meter.viz_dict())
    #         visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
    #                            xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper():  prec.top1()})
    # logger.debug('Val Acc: %f' % prec.top1())
    # return {'top1': prec.top1()}



def keep_relevant_outs(out, data, outputs):
    video_indexes = data['video_idx']
    t = data['t']
    rel_t = data['rel_t']
    times = data['times']
    video_names = data['video_name']
    for idx, video_idx in enumerate(video_indexes):
        o = out[idx]
        o = torch.softmax(o, dim=0)
        # if torch.argmax(o) != 1:
        #     continue

        if video_idx.item() not in outputs.keys():
            outputs[video_idx.item()] = {'time': torch.stack([t[0][idx], t[1][idx], t[2][idx]]),
                                         'rel_time': torch.stack([rel_t[0][idx], rel_t[1][idx], rel_t[2][idx]]),
                                         'clips': [{'confidence': o[1], 'f_time': (times[0][idx] + times[1][idx])/2 }],
                                         'video_name': video_names[idx]}
        else:
            outputs[video_idx.item()]['clips'].append({'confidence': o[1], 'f_time': (times[0][idx] + times[1][idx])/2})

    return outputs

def calc_acc(outs):
    total = 0
    correct = 0
    print(len(list(outs.keys())))
    best_for_vis = None
    worst_for_vis = None
    for key, value in outs.items():
        time = value['time']
        # time = torch.median(time)
        rel_t = value['rel_time']
        if not 0.01 <= torch.median(rel_t).item() <= 0.99:
            print('Outlier')
            continue
        clips = value['clips']
        max_conf = 0
        f_time = 0
        for clip in clips:
            if clip['confidence'].item() > max_conf:
                max_conf = clip['confidence'].item()
                f_time = clip['f_time']

        total += 1
        # if time.size() == 0:
        # if abs(f_time - time) <= 1:
        #     correct += 1
        # # else:
        if min(abs(f_time - t) for t in time) <= 1.0:
            if best_for_vis is None and 20 > len(clips) > 15:
                best_for_vis = {'video_name': value['video_name'], 'g_trn': time.mean(), 'p_trn': f_time}
            # else:
            #     if min(abs(f_time - t) for t in time) < abs(best_for_vis['g_trn'].item() - best_for_vis['p_trn']):
            #         t_idx = torch.argmin(torch.abs(time - f_time))
            #         best_for_vis = {'video_name': value['video_name'], 'g_trn': time[t_idx], 'p_trn': f_time}
            correct += 1
        else:
            if worst_for_vis is None and 1 < abs(f_time - time.mean()) < 1.5:
                worst_for_vis = {'video_name': value['video_name'], 'g_trn': time.mean(), 'p_trn': f_time}

    print(best_for_vis)
    print('Acc Val: %f' % (correct/total))

    # print(str(best_for_vis))
    # print(str(worst_for_vis))
    # visualize_perdiction(best_for_vis['video_name'], best_for_vis['g_trn'], best_for_vis['p_trn'])
    # visualize_perdiction(worst_for_vis['video_name'], worst_for_vis['g_trn'], worst_for_vis['p_trn'])
