# @Author: Enea Duka
# @Date: 5/7/21

import torch

from utils.util_functions import Precision
from tqdm import tqdm
from utils.logging_setup import viz
from utils.plotting_utils import visdom_plot_losses
from utils.util_functions import Meter
from utils.arg_parse import opt
from utils.logging_setup import logger
from transformer2x.trn_utils import prep_for_local, prep_for_global


def test(**kwargs):
    model = kwargs["model"]
    loss = kwargs["loss"]
    dataloader = kwargs["dataloader"]
    mode = kwargs["mode"]
    epoch = kwargs["epoch"]
    time_id = kwargs["time"]

    model.eval()
    prec = Precision(mode)
    meter = Meter(mode=mode, name="loss")
    bce_meter = Meter(mode=mode, name="bce_loss")

    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        for idx, data in iterator:
            keys = list(data.keys())
            videos = data.get(keys[0])
            # videos = videos.unsqueeze(1)
            labels = data["label"]
            pure_nr_frames = data["pure_nr_frames"]

            if opt.multi_scale:
                if labels.shape[0] == 1:
                    labels = labels.squeeze()
                else:
                    labels = labels.flatten()
                mask = labels != -1
                labels = labels[mask]
                labels = labels.type(torch.long)

                videos, position_ids, pnf = prep_for_local(videos, pure_nr_frames)
                if opt.backbone == "vit_longformer":
                    videos, _ = model(
                        videos,
                        position_ids,
                        None,
                        pnf,
                        labels,
                        local=True,
                        multi_scale=opt.multi_scale,
                    )
                else:
                    videos = model(
                        videos,
                        position_ids,
                        None,
                        pnf,
                        labels,
                        local=True,
                        multi_scale=opt.multi_scale,
                    )
                if len(pure_nr_frames.shape) == 2:
                    pure_nr_frames = torch.t(pure_nr_frames)[0]
                videos, position_ids, pure_nr_frames, num_clips = prep_for_global(
                    videos, position_ids, pure_nr_frames
                )
                out = model(
                    videos,
                    position_ids,
                    None,
                    pure_nr_frames,
                    labels,
                    num_clips,
                    False,
                    video_level_pred=True,
                )
            else:
                if opt.backbone == "vit_longformer":
                    position_ids = (
                        torch.tensor(list(range(0, videos.shape[1])))
                        .expand(1, videos.shape[1])
                        .repeat(videos.shape[0], 1)
                    )
                    out = model(videos, position_ids, None, pure_nr_frames)
                    out = out.squeeze()
                elif opt.backbone == "r3d_18":
                    out = model(videos.permute(0, 2, 1, 3, 4))

            _loss = loss(out, labels.cuda())
            meter.update(_loss.item(), videos.shape[0])

            if opt.task == "classification":
                prec.update_probs_sfx(out, labels.cuda())
            else:
                prec.update_probs_reg_rel(out, labels.cuda())

        meter.log()
        bce_meter.log()

        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(
                viz.env,
                opt.log_name + "-loss-" + str(time_id),
                epoch,
                xylabel=("epoch", "loss"),
                **meter.viz_dict()
            )
            visdom_plot_losses(
                viz.env,
                opt.log_name + "-prec-" + str(time_id),
                epoch,
                xylabel=("epoch", "prec@1"),
                **{"pr@1/%s" % mode.upper(): prec.top1()}
            )

    if opt.speed_and_motion:
        logger.debug("Epoch %d ==== Speed&Motoion Acc: %f" % (epoch, prec.top1()))
        return {"top1": prec.top1()}
    else:
        logger.debug("Val Acc: %f" % prec.top1())
