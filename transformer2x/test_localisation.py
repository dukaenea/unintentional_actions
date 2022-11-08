# @Author: Enea Duka
# @Date: 6/30/21


# @Author: Enea Duka
# @Date: 5/7/21

import torch

from utils.util_functions import Precision
from tqdm import tqdm
from utils.util_functions import Meter
from utils.arg_parse import opt
from utils.logging_setup import logger
from utils.util_functions import DistributionPlotter


def test_localisation(**kwargs):
    model = kwargs["model"]
    loss = kwargs["loss"]
    dataloader = kwargs["dataloader"]
    mode = kwargs["mode"]

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
        zeros = 0
        ones = 1
        twos = 2
        for idx, data in iterator:
            videos = data["features"].squeeze()
            pure_nr_frames = data["pure_nr_frames"].squeeze()
            # video_names = metadata['video_name']
            position_ids = (
                torch.tensor(list(range(0, videos.shape[1])))
                .expand(1, videos.shape[1])
                .repeat(videos.shape[0], 1)
            )
            out = model(videos, position_ids, None, pure_nr_frames)
            labels = data["label"].squeeze()
            # trn_times = metadata['t']
            # clip_boundries = metadata['clip_time_boarders'].squeeze()
            _loss = loss(out, labels.cuda())

            # if idx % 200 == 0 and idx > 0:
            #     logger.debug('Val Acc: %f' % prec.top1())
            for o in out:
                o = torch.softmax(o, dim=0)
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
        print(
            "Class predictions: %f, %f, %f"
            % (zeros / total, ones / total, twos / total)
        )
        print(total)
        meter.log()
        logger.debug("Val Acc: %f" % prec.top1(report_pca=True))
        calc_acc(outputs)


def keep_relevant_outs(out, data, outputs):
    video_indexes = data["video_idx"]
    t = data["t"]
    rel_t = data["rel_t"]
    times = data["times"]
    video_names = data["video_name"]
    for idx, video_idx in enumerate(video_indexes):
        o = out[idx]
        o = torch.softmax(o, dim=0)
        # if torch.argmax(o) != 1:
        #     continue

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
    correct = 0
    print(len(list(outs.keys())))
    best_for_vis = None
    worst_for_vis = None
    for key, value in outs.items():
        time = value["time"]
        # time = torch.median(time)
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
        if min(abs(f_time - t) for t in time) <= 0.25:
            if best_for_vis is None and 20 > len(clips) > 15:
                best_for_vis = {
                    "video_name": value["video_name"],
                    "g_trn": time.mean(),
                    "p_trn": f_time,
                }
            correct += 1
        else:
            if worst_for_vis is None and 1 < abs(f_time - time.mean()) < 1.5:
                worst_for_vis = {
                    "video_name": value["video_name"],
                    "g_trn": time.mean(),
                    "p_trn": f_time,
                }

    print(best_for_vis)
    print("Acc Val: %f" % (correct / total))
