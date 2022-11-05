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
from transformer2x.trn_utils import prep_for_local, prep_for_global, prep_for_crf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import os
import csv
import cv2
from visualization.tsne_visualizer import do_tsne


vis_path = "/BS/unintentional_actions/work/storage/visualisations"
vis_path_plot = "/BS/unintentional_actions/work/storage/visualisations/plots"
base_vid_path = "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val"


def test(**kwargs):
    model = kwargs["model"]
    feat_extractor = kwargs["feat_extractor"]
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
    best_scores = [0, 0, 0]
    best_lengths = [0, 0, 0]
    best_out = None
    best_t = None
    best_times = None
    best_vid_name = None

    video_names = []
    # with open(os.path.join(vis_path, 'vis_meta.csv'), 'r') as f:
    #     csvreader = csv.DictReader(f)
    #     line_count = 0
    #     for row in csvreader:
    #         video_names.append(row['video_name'])
    all_outs = []
    all_labels = []
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
            if not opt.use_crf:
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
            if opt.multi_scale:
                videos, position_ids, pure_nr_frames, num_clips = prep_for_global(
                    videos, position_ids, pure_nr_frames
                )
                out = model(
                    videos,
                    position_ids,
                    None,
                    pure_nr_frames,
                    labels,
                    local=False,
                    multi_scale=opt.multi_scale,
                    return_features=False,
                )
            else:
                num_clips = pure_nr_frames.shape[0]
                out = videos

            all_outs.extend(list(out.detach().cpu()))
            all_labels.extend(labels)

            if opt.use_crf:
                videos = prep_for_crf(out, pure_nr_frames)

                if opt.crf_margin_probs:
                    probs = model(
                        videos,
                        position_ids,
                        None,
                        pure_nr_frames,
                        labels,
                        num_clips,
                        False,
                        True,
                    )
                    probs = probs.reshape(
                        probs.shape[0] * probs.shape[1], probs.shape[2]
                    )
                    labels = labels.flatten()
                    mask = labels != -1
                    labels = labels[mask]
                    probs = probs[mask]
                    _loss = loss(probs, labels.cuda())
                else:
                    out, _loss = model(
                        videos, None, None, pure_nr_frames, labels, None, False, True
                    )
                    _loss = _loss.mean()
                    labels = labels.flatten()
                    mask = labels != -1
                    labels = labels[mask]
                    labels = labels.type(torch.long)
            else:
                _loss = loss(out, labels.cuda())

            meter.update(_loss.item(), videos.shape[0])

            if opt.use_crf:
                if opt.crf_margin_probs:
                    prec.update_probs_sfx(
                        probs, labels.cuda(), report_pca=True, num_classes=3
                    )
                else:
                    prec.update_probs_crf(
                        out, labels.cuda(), report_pca=True, num_classes=3
                    )
            else:
                prec.update_probs_sfx(
                    out, labels.cuda(), report_pca=True, num_classes=3
                )

            visualize_crf_preds(out, pure_nr_frames, data, video_names, labels)
            # best_scores, best_lengths, best_out, best_t, best_times = keep_cleanest_video(out, labels, best_scores,
            #                                                                               best_lengths, pure_nr_frames,
            #                                                                               best_out, data, best_t,
            #                                                                               best_times, vis_metadata)
            # outputs = keep_relevant_outs(out, data, outputs)

        # print(total)
        # print(best_scores)
        # print(best_lengths)
        # print(best_t)
        # print(best_times)
        # print(best_out)
        # write_vis_meta_to_csv(vis_metadata)
        # all_outs = [o.numpy() for o in all_outs]
        # all_labels = [l.item() for l in all_labels]
        # all_labels = torch.tensor(all_labels)
        # do_tsne(all_outs, all_labels, 'Feature Space (F2C2V)')

        meter.log()
        # plot_scores(best_out, best_times, best_t)
        # calc_acc(outputs)
        # if opt.task == 'regression':
        #     out_dist_plotter.plot_out_dist()
        if opt.viz and epoch % opt.viz_freq == 0:
            visdom_plot_losses(
                viz.env,
                opt.log_name + "-loss-" + str(time_id),
                epoch,
                xylabel=("epoch", "loss"),
                **meter.viz_dict()
            )
            # if not opt.mmargin_loss:
            visdom_plot_losses(
                viz.env,
                opt.log_name + "-prec-" + str(time_id),
                epoch,
                xylabel=("epoch", "prec@1"),
                **{"pr@1/%s" % mode.upper(): prec.top1()}
            )
    # if not opt.mmargin_loss:
    logger.debug("Epoch %d Val Acc: %f" % (epoch, prec.top1(report_pca=True)))
    # acc_t = torch.Tensor([prec.lab_class[0]['correct']/prec.lab_class[0]['total'],
    #                     prec.lab_class[1]['correct']/prec.lab_class[1]['total'],
    #                     prec.lab_class[2]['correct']/prec.lab_class[2]['total']])
    return {
        "top1": (1 / meter.avg) if opt.mmargin_loss else prec.top1()
    }  # ,  torch.softmax(acc_t, dim=0)
    # return {'top1': prec.top1()}


def visualize_crf_preds(outs, pnfs, data, video_names, labels):
    for idx, pnf in enumerate(pnfs):
        video_name = data["video_name"][idx]
        if video_name in video_names:
            if idx == 0:
                scores = outs[: int(pnf.item())]
                l = labels[: int(pnf.item())]
            else:
                scores = outs[
                    int(pnfs[:idx].sum().item()) : int(pnfs[:idx].sum().item())
                    + int(pnf.item())
                ]
                l = labels[
                    int(pnfs[:idx].sum().item()) : int(pnfs[:idx].sum().item())
                    + int(pnf.item())
                ]

            x = list(range(scores.shape[0]))
            y = scores.detach().cpu().numpy()
            y_l = l.detach().cpu().numpy()

            scatter_points_colors = list(
                map(lambda i: "green" if i == 0 else ("yellow" if i == 1 else "red"), y)
            )
            plt.scatter(x, y, marker="o", linewidths=0.5, s=30, c=scatter_points_colors)
            plt.scatter(x, y_l, marker="x", s=30, linewidths=0.5)
            plt.axis("equal")
            plt.axis("off")
            plt.savefig(
                os.path.join(vis_path_plot, video_name + "_crf.svg"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=1000,
            )
            plt.clf()

            extract_video_frames(video_name)

        continue


def extract_video_frames(video_name):
    video_path = os.path.join(base_vid_path, video_name + ".mp4")
    cam = cv2.VideoCapture(video_path)
    frame_folder = os.path.join(vis_path, "frames", video_name)
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

    current_frame = 0

    while True:
        ret, frame = cam.read()
        if ret:
            frame_path = os.path.join(frame_folder, str(current_frame) + ".jpg")
            cv2.imwrite(frame_path, frame)
            current_frame += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


def write_vis_meta_to_csv(vis_meta):
    with open(os.path.join(vis_path, "vis_meta.csv"), "w", newline="") as f:
        dict_writer = csv.DictWriter(f, vis_meta[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(vis_meta)


def keep_cleanest_video(
    outs,
    labels,
    best_scores,
    best_lengths,
    pnfs,
    best_out,
    batch,
    best_t,
    best_times,
    vis_metadata,
):
    for idx, pnf in enumerate(pnfs):

        if idx == 0:
            l = labels[: int(pnf.item())]
            scores = outs[: int(pnf.item())]
        else:
            l = labels[
                int(pnfs[:idx].sum().item()) : int(pnfs[:idx].sum().item())
                + int(pnf.item())
            ]
            scores = outs[
                int(pnfs[:idx].sum().item()) : int(pnfs[:idx].sum().item())
                + int(pnf.item())
            ]

        scores = torch.softmax(scores, dim=1)
        zeros_idc = l == 0
        ones_idc = l == 1
        twos_idc = l == 2

        out_zeros = scores[zeros_idc]
        out_ones = scores[ones_idc]
        out_twos = scores[twos_idc]

        if out_twos.shape[0] == 0:
            continue

        mean_zeros = out_zeros[:, 0].mean()
        mean_ones = out_ones[:, 1].mean()
        mean_twos = out_twos[:, 2].mean()
        lengths = [out_zeros.shape[0], out_ones.shape[0], out_twos.shape[0]]

        if (
            mean_zeros > best_scores[0]
            and mean_ones > best_scores[1]
            and mean_twos > best_scores[2]
            and mean_zeros < 0.75
            and mean_ones < 0.75
            and mean_twos < 0.75
        ):
            # best_scores = [mean_zeros, mean_ones, mean_twos]
            # best_lengths = lengths
            # best_out = scores
            # best_t = batch['t'][1][idx]
            # best_times = batch['times'][idx]
            clip_times = plot_scores(
                scores,
                batch["times"][idx],
                batch["t"][1][idx],
                batch["video_name"][idx],
            )
            print(batch["video_name"][idx])
            vis_metadata.append(
                {"video_name": batch["video_name"][idx], "clip_times": clip_times}
            )
            break

    return best_scores, best_lengths, best_out, best_t, best_times


def keep_relevant_outs(out, data, outputs):
    video_indexes = data["video_idx"]
    t = data["t"]
    rel_t = data["rel_t"]
    times = data["times"]
    times = "".join([item for sublist in times for item in sublist]).split("~")
    pnfs = data["pure_nr_frames"].permute(1, 0)
    pnfs = pnfs[0]
    # times = times.split('~')
    video_names = data["video_name"]
    # vid_idx = []
    # for indx, pnf in zip(video_indexes, pnfs):
    #     ind = [indx.item()] * int(pnf.item())
    #     vid_idx.extend(ind)
    # video_indexes = torch.tensor(vid_idx)
    for idx, video_idx in enumerate(video_indexes):
        # if torch.argmax(o) != 1:
        #     continue
        # video_times = times[idx]
        # video_times = video_times.split('~')
        # video_times = video_times[:-1]
        start_idx = int(torch.sum(pnfs[:idx]).item())
        end_idx = int(torch.sum(pnfs[: idx + 1]).item())
        for i in range(start_idx, end_idx):
            o = out[i]
            o = torch.softmax(o, dim=0)
            try:
                if video_idx.item() not in outputs.keys():
                    outputs[video_idx.item()] = {
                        "time": torch.stack([t[0][idx], t[1][idx], t[2][idx]]),
                        "rel_time": torch.stack(
                            [rel_t[0][idx], rel_t[1][idx], rel_t[2][idx]]
                        ),
                        "clips": [
                            {
                                "confidence": o[1],
                                "f_time": (
                                    float(times[i * 2]) + float(times[i * 2 + 1])
                                )
                                / 2,
                            }
                        ],
                        "video_name": video_idx.item(),
                    }
                else:
                    outputs[video_idx.item()]["clips"].append(
                        {
                            "confidence": o[1],
                            "f_time": (float(times[i * 2]) + float(times[i * 2 + 1]))
                            / 2,
                        }
                    )
            except Exception:
                print("here")
    return outputs


def calc_acc(outs):
    total = 0
    correct_tf = 0
    correct_one = 0
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
        if min(abs(f_time - t) for t in time) <= 1:
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
            correct_one += 1
        else:
            if worst_for_vis is None and 1 < abs(f_time - time.mean()) < 1.5:
                worst_for_vis = {
                    "video_name": value["video_name"],
                    "g_trn": time.mean(),
                    "p_trn": f_time,
                }

        if min(abs(f_time - t) for t in time) <= 0.25:
            correct_tf += 1

    print(best_for_vis)
    print("Acc Val 0.25: %f" % (correct_tf / total))
    print("Acc Val 1.00: %f" % (correct_one / total))


def plot_scores(scores, times, video_name):
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
    # plt.legend(['Intentional', 'Transitional', 'Unintentional'], loc='lower right')
    # plt.show()
    # f = interp1d(x, trn_cnf, kind='quadratic')
    # trn_cnf = f(x)
    #
    # plt.plot(x, nrm_cnf, lw=1, color='#61c46e')
    # plt.fill_between(x, 0, nrm_cnf, alpha=0.3, facecolor='#61c46e')
    # plt.plot(x, trn_cnf, lw=1, color='#f6ed22')
    # plt.fill_between(x, 0, trn_cnf, alpha=0.3, facecolor='#f6ed22')
    # plt.plot(x, unrm_cnf, lw=1, color='#f15a24')
    # plt.fill_between(x, 0, unrm_cnf, alpha=0.3, facecolor='#f15a24')
    # plt.axvline(x=t)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Prediction confidence')
    # plt.show()

    return x


import pandas as pd
import json

if __name__ == "__main__":
    paths = [
        "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Are You Serious! - Throwback Thursday (September 2017) _ FailArmy64.mp4",
        "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Best Fails of the Year 2017 - Part 1 (December 2017) _ FailArmy12.mp4",
        "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Fails of the Week - Insult to Injury (January 2017) _ FailArmy1.mp4",
        "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Let's Get It!! - FailArmy After Dark (ep. 2)88.mp4",
        "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Funny School Fails Compilation _ 'School's Out' By FailArmy 20167.mp4",
        "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Break Yourself - Fails of the Week (September 2017) _ FailArmy15.mp4",
    ]

    with open(
        "/BS/unintentional_actions/work/data/oops/epstain/results_-00001_016.json"
    ) as f:
        preds = json.load(f)

    pred_conf = []
    times = []
    for idx, pred in enumerate(preds):
        path = pred["fn"]
        if path in paths:
            conf = torch.tensor(pred["y_hat_vec"])
            time = (pred["t_end"] - pred["t_start"]) / 2 + pred["t_start"]

            pred_conf.append(conf)
            times.append(time)
        else:
            if len(pred_conf) > 0:
                path = preds[idx - 1]["fn"]
                name_parts = path.split("/")
                name = name_parts[-1]
                name = name[:-4]
                plot_scores(torch.stack(pred_conf), times, name)
                pass
            pred_conf = []
            times = []
