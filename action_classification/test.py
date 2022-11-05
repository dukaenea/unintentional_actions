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
from utils.util_functions import DistributionPlotter
from transformer2x.trn_utils import prep_for_local, prep_for_crf


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
    outputs = {}

    with torch.no_grad():
        if opt.use_tqdm:
            iterator = enumerate(tqdm(dataloader))
        else:
            iterator = enumerate(dataloader)
        total = 0
        for idx, data in iterator:
            videos = data["features"]
            pure_nr_frames = data["pure_nr_frames"]
            labels = data["label"]

            if opt.use_crf:
                videos, position_ids, pnf = prep_for_local(videos, pure_nr_frames)
                if opt.backbone == "vit_longformer":
                    out = model(videos, position_ids, None, pure_nr_frames)
                else:
                    out = model(videos.permute(0, 2, 1, 3, 4))
                pure_nr_frames = torch.t(pure_nr_frames)[0]
                videos = prep_for_crf(out, pure_nr_frames)
                if opt.backbone == "vit_longformer":
                    out, _loss = model(
                        videos, None, None, pure_nr_frames, labels=labels, for_crf=True
                    )
                else:
                    out, _loss = model(videos, labels=labels, for_crf=True)
                _loss = _loss.mean()
                labels = labels.flatten()
                mask = labels != -1
                labels = labels[mask]
                labels = labels.type(torch.long)
            else:
                if opt.backbone == "vit_longformer":
                    position_ids = (
                        torch.tensor(list(range(0, videos.shape[1])))
                        .expand(1, videos.shape[1])
                        .repeat(videos.shape[0], 1)
                    )
                    out = model(
                        videos,
                        position_ids,
                        None,
                        pure_nr_frames,
                        classifier_only=False,
                    )
                else:
                    out = model(videos)

                _loss = loss(out, labels.cuda())

            meter.update(_loss.item(), videos.shape[0])
            if opt.use_crf:
                prec.update_probs_crf(out, labels.cuda())
            else:
                prec.update_probs_sfx(
                    out, labels.cuda(), report_pca=False, num_classes=3
                )

            outputs = keep_relevant_outs(out, data, outputs)
        print(total)
        meter.log()
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
                **{
                    "pr@1/%s" % mode.upper(): prec.top1(),
                    "(0)pr@1/%s"
                    % mode.upper(): prec.lab_class[0]["correct"]
                    / prec.lab_class[0]["total"],
                    "(1)pr@1/%s"
                    % mode.upper(): prec.lab_class[1]["correct"]
                    / prec.lab_class[1]["total"],
                    "(2)pr@1/%s"
                    % mode.upper(): prec.lab_class[2]["correct"]
                    / prec.lab_class[2]["total"],
                }
            )
        calc_acc(outputs)
    logger.debug("Val Acc: %f" % prec.top1(report_pca=False))
    return {"top1": prec.top1()}


def keep_relevant_outs(out, data, outputs):
    video_indexes = data["video_idx"]
    t = data["t"]
    rel_t = data["rel_t"]
    times = data["times"]
    video_names = data["video_name"]
    for idx, video_idx in enumerate(video_indexes):
        o = out[idx]
        o = torch.softmax(o, dim=0)

        if video_idx.item() not in outputs.keys():
            outputs[video_idx.item()] = {
                "time": torch.stack([t[0][idx], t[1][idx], t[2][idx]]),
                "rel_time": torch.stack([rel_t[0][idx], rel_t[1][idx], rel_t[2][idx]]),
                "clips": [
                    {"confidence": o[1], "f_time": (times[0][idx] + times[1][idx]) / 2}
                ],
                "video_name": video_names[idx],
            }
        else:
            outputs[video_idx.item()]["clips"].append(
                {"confidence": o[1], "f_time": (times[0][idx] + times[1][idx]) / 2}
            )

    return outputs


def calc_acc(outs):
    total = 0
    correct_one = 0
    correct_tf = 0
    accs = []
    print(len(list(outs.keys())))
    best_for_vis = None
    worst_for_vis = None
    for key, value in outs.items():
        time = value["time"]
        rel_t = value["rel_time"]
        if not 0.01 <= torch.median(rel_t).item() <= 0.99:
            print("Outlier")
            continue
        clips = value["clips"]
        max_conf = 0
        f_time = 0
        for clip in clips:
            if clip["confidence"].item() > max_conf:
                max_conf = clip["confidence"].item()
                f_time = clip["f_time"]

        total += 1
        acc_cls = (100 * (min(abs(f_time - t) for t in time) <= 1.0)).item()
        accs.append(acc_cls)
        if min(abs(f_time - t) for t in time) <= 0.25:
            if best_for_vis is None and 20 > len(clips) > 15:
                best_for_vis = {
                    "video_name": value["video_name"],
                    "g_trn": time.mean(),
                    "p_trn": f_time,
                }
            # else:
            #     if min(abs(f_time - t) for t in time) < abs(best_for_vis['g_trn'].item() - best_for_vis['p_trn']):
            #         t_idx = torch.argmin(torch.abs(time - f_time))
            #         best_for_vis = {'video_name': value['video_name'], 'g_trn': time[t_idx], 'p_trn': f_time}
            correct_tf += 1
        else:
            if worst_for_vis is None and 1 < abs(f_time - time.mean()) < 1.5:
                worst_for_vis = {
                    "video_name": value["video_name"],
                    "g_trn": time.mean(),
                    "p_trn": f_time,
                }

        if min(abs(f_time - t) for t in time) <= 1.0:
            correct_one += 1

    print(best_for_vis)
    print("Acc Val 0.25: %f" % (correct_tf / total))
    print("Acc Val 1.00: %f" % (correct_one / total))
