# @Author: Enea Duka
# @Date: 5/3/21
import sys
import os
from dataloaders.kinetics_loader import KineticsDataset
from transformer2x.my_oops_loader import get_video_loader_trn_2x
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
from transformer2x.vtn import create_model_trn_2x
from transformer2x.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime
from torch.utils.data.dataloader import default_collate

import matplotlib.pyplot as plt
import numpy as np
import cv2


def do_uar():
    opt.dataset = "oops"
    opt.rep_data_level = "features"
    opt.rep_backbone = "resnet"
    opt.lr_scheduler = "step"
    opt.backbone = "vit_longformer"

    opt.embed_dim = 512
    opt.intermediate_size = 3072
    opt.hidden_dim = 512
    opt.mlp_dim = 1024
    opt.num_classes = 7
    opt.num_classes_ptr = 7

    # pretrained on contrastive loss
    opt.model_name = "VideoLongformer_MLP"
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 128.0
    opt.sfx = str(
        "%s.unint_act.%s.layers%d.attn_win%d_2x_trn"
        % (opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0])
    )
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False
    opt.use_bbone = False
    opt.mmargin_loss = False
    opt.use_crf = True
    opt.crf_margin_probs = False
    opt.use_frame_encoder = True

    opt.optim = "adam"
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.01
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1
    opt.multi_scale = False
    pretrained = True  #################################################################################################
    opt.log_name = "lr:%f~ep:%d~bs:%d~win:%d~b_lr:%f~ptr:%s_cntr_loss" % (
        opt.lr,
        opt.epochs,
        opt.batch_size,
        opt.attention_window[0],
        opt.backbone_lr_factor,
        str(pretrained),
    )
    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
    )
    opt.sfx = str(
        "%s.unint_act.%s.layers%d.attn_win%d.time%s_trn_2x"
        % (
            opt.dataset,
            opt.task,
            opt.num_hidden_layers,
            opt.attention_window[0],
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )

    opt.debug = False

    if opt.debug:
        opt.num_workers = 0
        opt.batch_size = 2
        opt.save_model = False
        opt.epochs = 1

        opt.viz = False

    setup_logger_path()

    train_set, val_set, test_set = None, None, None
    if opt.dataset == "kinetics":
        train_set = KineticsDataset(
            "train",
            fps=25,
            fpc=32,
            spat_crop=True,
            hflip=False,
            norm_statistics={
                "mean": [0.43216, 0.394666, 0.37645],
                "std": [0.22803, 0.22145, 0.216989],
            },
            feat_ext=True,
            data_level=opt.rep_data_level,
            feat_set="%s_feats" % opt.rep_backbone,
        )
        val_set = KineticsDataset(
            "val",
            fps=25,
            fpc=32,
            spat_crop=True,
            hflip=False,
            norm_statistics={
                "mean": [0.43216, 0.394666, 0.37645],
                "std": [0.22803, 0.22145, 0.216989],
            },
            feat_ext=True,
            data_level=opt.rep_data_level,
            feat_set="%s_feats" % opt.rep_backbone,
        )
        test_set = KineticsDataset(
            "test",
            fps=25,
            fpc=32,
            spat_crop=True,
            hflip=False,
            norm_statistics={
                "mean": [0.43216, 0.394666, 0.37645],
                "std": [0.22803, 0.22145, 0.216989],
            },
            feat_ext=True,
            data_level=opt.rep_data_level,
            feat_set="%s_feats" % opt.rep_backbone,
        )
    elif opt.dataset == "rareact":
        pass
    elif opt.dataset == "oops":
        train_set = SimpleOopsDataset(
            "train",
            16,
            "features" if opt.task == "classification" else "frames",
            True,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            balance=True,
        )
        val_set = SimpleOopsDataset(
            "val",
            16,
            "features" if opt.task == "classification" else "frames",
            True,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
        pass
    elif opt.dataset == "all":
        pass

    if opt.dataset == "oops":
        opt.batch_size = 8
        opt.workers = 16
        opt.balance_fails_only = False
        opt.all_fail_videos = False
        opt.load_videos = True
        opt.load_frames = opt.backbone == "r3d_18"
        opt.step_between_clips_sec = 1
        # opt.anticipate_label = 0
        train_loader = get_video_loader_trn_2x(opt)
        opt.val = True
        opt.fails_path = (
            "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
        )
        val_loader = get_video_loader_trn_2x(opt)

    else:
        train_loader = DataLoader(
            train_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_set.pad_videos_collate_fn
            if opt.task == "regression"
            else default_collate,
        )

        val_loader = DataLoader(
            val_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=train_set.pad_videos_collate_fn
            if opt.task == "regression"
            else default_collate,
        )

    model, optimizer, loss = create_model_trn_2x(
        num_classes=100 if opt.dataset == "kinetics" else 3, pretrained=pretrained
    )

    # if pretrained:
    #     optimizer = get_froze_trn_optimizer(model)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epochs=25,
    )


def plot_qual_results(nrm_cnf, trn_cnf, unrm_crf, vid_path):
    def _frame_image(img, frame_width, frame_color):
        b = frame_width  # border size in pixel
        ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y
        if img.ndim == 3:  # rgb or rgba array
            framed_img = np.zeros((b + ny + b, b + nx + b, img.shape[2]))
            framed_img[:, :, 0] = frame_color[0]
            framed_img[:, :, 1] = frame_color[1]
            framed_img[:, :, 2] = frame_color[2]
        elif img.ndim == 2:  # grayscale image
            framed_img = np.zeros((b + ny + b, b + nx + b))
        framed_img[b:-b, b:-b] = img
        framed_img = framed_img.astype("int")
        return framed_img

    save_path = "/BS/unintentional_actions/work/unintentional_actions"
    x = list(range(len(nrm_cnf)))
    video = cv2.VideoCapture(vid_path)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    nr_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    nr_clips = nr_frames // 16
    idc = list(range(0, nr_frames, 16))
    idc = [elm + 8 for elm in idc]
    counter = 0
    curr_idc_idx = 0
    cols = len(idc)
    rows = 1

    fig, ax = plt.subplots(rows, cols, figsize=(10, 1))
    while True:
        ret, frame = video.read()
        if counter == idc[curr_idc_idx]:
            ax[curr_idc_idx].imshow(_frame_image(frame, int(100), [97, 196, 110]))
            ax[curr_idc_idx].axis("off")
            curr_idc_idx += 1
            if curr_idc_idx >= len(idc):
                break
        counter += 1
    plt.axis("off")
    # plt.show()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(save_path, "fig.png"), dpi=1000)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0 1 2 3'
    do_uar()
