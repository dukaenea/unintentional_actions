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
from visualization.visualize_unint_action_localization import visualize_perdiction
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
from visualization.tsne_visualizer import do_tsne

vis_path_plot = "/BS/unintentional_actions/work/storage/visualisations/plots"
vis_video_names = [
    "Are You Serious! - Throwback Thursday (September 2017) _ FailArmy64",
    "Best Fails of the Year 2017 - Part 1 (December 2017) _ FailArmy12",
    "Fails of the Week - Insult to Injury (January 2017) _ FailArmy1",
    "Let's Get It!! - FailArmy After Dark (ep. 2)88",
    "Funny School Fails Compilation _ 'School's Out' By FailArmy 20167",
    "Break Yourself - Fails of the Week (September 2017) _ FailArmy15",
]


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
    if opt.task == "regression":
        out_dist_plotter = DistributionPlotter()

    outputs = {}
    vis_dict = {}

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
            videos = data["features"].squeeze()
            pure_nr_frames = data["pure_nr_frames"].squeeze()
            # video_names = data['video_name']
            position_ids = (
                torch.tensor(list(range(0, videos.shape[1])))
                .expand(1, videos.shape[1])
                .repeat(videos.shape[0], 1)
            )
            out = model(
                videos, position_ids, None, pure_nr_frames, return_features=False
            )
            labels = data["label"].squeeze()
            all_outs.extend(list(out.detach().cpu()))
            all_labels.extend(labels)
            _loss = loss(out, labels.cuda())

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
            vis_dict = gather_vis_stats(
                out, data["times"], data["video_name"], vis_dict
            )

            meter.update(_loss.item(), videos.shape[0])
            prec.update_probs_sfx(out, labels.cuda(), report_pca=True, num_classes=3)

            outputs = keep_relevant_outs(out, data, outputs)
        # all_outs = [o.numpy() for o in all_outs]
        # all_labels = [l.item() for l in all_labels]
        # all_labels = torch.tensor(all_labels)
        # do_tsne(all_outs, all_labels, 'Feature Space (F2C) + TTIBUA')

        print(
            "Class predictions: %f, %f, %f"
            % (zeros / total, ones / total, twos / total)
        )
        print(total)
        meter.log()
        logger.debug("Val Acc: %f" % prec.top1(report_pca=True))
        calc_acc(outputs)
        plot_vis_dicts(vis_dict)


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
    accs = []
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
        # if time.size() == 0:
        # if abs(f_time - time) <= 1:
        #     correct += 1
        # # else:
        acc_cls = (100 * (min(abs(f_time - t) for t in time) <= 1.0)).item()
        accs.append(acc_cls)
        if min(abs(f_time - t) for t in time) <= 1.0:
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
    print(
        "mean accuracy: {0:4} +- {1:4}".format(
            statistics.mean(accs), statistics.stdev(accs)
        )
    )


def plot_vis_dicts(vis_dict):
    for key, value in vis_dict.items():
        # sort the value according to the time
        times = value["times"]
        sort_idxs = np.argsort(value["times"])
        # times = np.array(value['times'])
        # times = times[sort_idxs]
        scores = torch.stack(value["scores"], dim=0)

        plot_video_scores(scores, times, key)


def gather_vis_stats(scores, times, video_names, vis_dict):
    for i in range(scores.shape[0]):
        score = scores[i]
        start_time = times[0][i]
        end_time = times[1][i]
        video_name = video_names[i]

        if video_name in vis_video_names:
            score = torch.softmax(score, dim=0)
            time = (start_time + end_time) / 2
            if video_name in vis_dict:
                vis_dict[video_name]["scores"].append(score)
                vis_dict[video_name]["times"].append(time.item())
            else:
                vis_dict[video_name] = {}
                vis_dict[video_name]["scores"] = [score]
                vis_dict[video_name]["times"] = [time.item()]

    return vis_dict


def plot_video_scores(scores, times, video_name):
    scatter_points = torch.max(scores, dim=1)
    scatter_points_colors = list(
        map(
            lambda i: "green" if i == 0 else ("yellow" if i == 1 else "red"),
            scatter_points[1],
        )
    )
    scatter_points = scatter_points[0].detach().cpu().numpy()
    scores = torch.t(scores)
    nrm_cnf = scores[0].detach().cpu().numpy()
    trn_cnf = scores[1].detach().cpu().numpy()
    unrm_cnf = scores[2].detach().cpu().numpy()

    # times = ''.join([item for sublist in times for item in sublist]).split('~')
    # times = [float(elm) for elm in times[:-1]]
    # x = [round((a + b) / 2, 2) for a, b in zip(times[0::2], times[1::2])]
    x = [round(a, 2) for a in times]
    x_intp = np.linspace(min(x), max(x), 500)

    f_n = interp1d(x, nrm_cnf, kind="linear")
    nrm_cnf = f_n(x_intp)
    nrm_cnf[nrm_cnf < 0] = 0

    f_t = interp1d(x, trn_cnf, kind="linear")
    trn_cnf = f_t(x_intp)
    trn_cnf[trn_cnf < 0] = 0

    f_u = interp1d(x, unrm_cnf, kind="linear")
    unrm_cnf = f_u(x_intp)
    unrm_cnf[unrm_cnf < 0] = 0

    # plt.figure(num=None, figsize=(3, 1))
    plt.plot(x_intp, nrm_cnf, lw=1, color="#61c46e")
    plt.fill_between(x_intp, 0, nrm_cnf, alpha=0.3, facecolor="#61c46e")
    plt.plot(x_intp, trn_cnf, lw=1, color="#f6ed22")
    plt.fill_between(x_intp, 0, trn_cnf, alpha=0.3, facecolor="#f6ed22")
    plt.plot(x_intp, unrm_cnf, lw=1, color="#f15a24")
    plt.fill_between(x_intp, 0, unrm_cnf, alpha=0.3, facecolor="#f15a24")
    plt.scatter(
        x, scatter_points, marker="o", linewidths=0.5, s=10, c=scatter_points_colors
    )
    # plt.axvline(x=t)
    plt.axis("equal")
    plt.ylim(0, 1)
    plt.xlabel("Time (sec)")
    plt.ylabel("Prediction confidence")
    plt.axis("off")
    plt.savefig(
        os.path.join(vis_path_plot, video_name + ".svg"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=1000,
    )
    plt.clf()

    return x
