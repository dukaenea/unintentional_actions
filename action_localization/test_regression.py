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


def test(**kwargs):
    model = kwargs['model']
    feat_extractor = kwargs['feat_extractor']
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

    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        total = 0
        for idx, data in iterator:
            videos = data['features']
            pure_nr_frames = data['pure_nr_frames']
            position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                .expand(1, videos.shape[1]) \
                .repeat(videos.shape[0], 1)
            out = model(videos, position_ids, None, pure_nr_frames)
            if opt.task == 'classification':
                labels = data['label']
                _loss = loss(out, labels.cuda())
            elif opt.task == 'regression':
                out = out.permute(1, 0).squeeze()
                # out_dist_plotter.update_bins(out)
                abs_t = data['t']
                labels = data['rel_t']
                lengths = data['len']
                _loss = loss(out, labels.cuda())


            meter.update(_loss.item(), videos.shape[0])

            if opt.task == 'classification':
                prec.update_probs_sfx(out, labels.cuda())
            elif opt.task == 'regression':
                prec.update_probs_reg(out, abs_t.cuda(), lengths.cuda())
        print(total)
        meter.log()
        # if opt.task == 'regression':
        #     out_dist_plotter.plot_out_dist()
        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
                               xylabel=('epoch', 'loss'), **meter.viz_dict())
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
                               xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper():  prec.top1()})
    logger.debug('Val Acc: %f' % prec.top1())
    return {'top1': prec.top1()}
