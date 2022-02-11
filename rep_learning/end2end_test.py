
# @Author: Enea Duka
# @Date: 9/1/21

from utils.util_functions import Precision
from utils.util_functions import Meter
from utils.arg_parse import opt
from tqdm import tqdm
from utils.logging_setup import viz
from utils.plotting_utils import visdom_plot_losses
from utils.logging_setup import logger
import torch


def test(**kwargs):
    model = kwargs['model']
    feat_extractor = kwargs['feat_extractor']
    loss = kwargs['loss']
    dataloader = kwargs['dataloader']
    mode = kwargs['mode']
    epoch = kwargs['epoch']
    time_id = kwargs['time']

    model.eval()
    feat_extractor.eval()

    prec = Precision(mode)
    loss_meter = Meter(mode=mode, name='loss')
    scaler = torch.cuda.amp.GradScaler()
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
            with torch.cuda.amp.autocast():
                # videos = extract_features(videos, feat_extractor)

                position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
                    .expand(1, videos.shape[1]) \
                    .repeat(videos.shape[0], 1)
                # start_time = time.time()
                out = model(videos, position_ids, None, pure_nr_frames)

                _loss = loss(out, labels.cuda())
            loss_meter.update(_loss.item(), videos.shape[0])
            prec.update_probs_sfx(out, labels.cuda())
            # if idx == 20:
            #     break
        loss_meter.log()

        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
                               xylabel=('epoch', 'loss'), **loss_meter.viz_dict())
            # if not opt.mmargin_loss:
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
                               xylabel=('epoch', 'prec@1'), **{
                    'pr@1/%s' % mode.upper(): prec.top1(),
                    '(0)pr@1/%s' % mode.upper(): prec.lab_class[0]['correct'] / prec.lab_class[0]['total'],
                    '(1)pr@1/%s' % mode.upper(): prec.lab_class[1]['correct'] / prec.lab_class[1]['total'],
                    '(2)pr@1/%s' % mode.upper(): prec.lab_class[2]['correct'] / prec.lab_class[2]['total'],
                })
    # if not opt.mmargin_loss:
    logger.debug('Val Acc: %f' % prec.top1(report_pca=True))
    return {'top1': prec.top1()}




def extract_features(clips, feat_extractor):
    # start = time.time()
    c, f, ch, w, h = clips.shape
    # rearrange the batch
    # clips = clips.permute(0, 2, 1, 3, 4)
    # concat all the clips of the videos
    clips = clips.reshape(c*f, ch, w, h)
    clips = feat_extractor(clips.cuda())

    clips = torch.stack(list(clips.split(f)))
    # total_time = time.time() - start
    # logger.debug("Time to extract the features: %f" % total_time)
    return clips


