# @Author: Enea Duka
# @Date: 5/3/21
import sys
import warnings

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")

from dataloaders.my_oops_loader import get_video_loader
from dataloaders.oops_loader import get_video_loader_frames
from transformer2x.my_oops_loader import get_video_loader_trn_2x
from models.pm_vtn import create_model
from models.res3d_18 import create_r3d
from action_classification.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime


def do_uar():

    opt.log_name = "lr:%f~ep:%d~bs:%d~win:%d~b_lr:%f~ptr:%s_cntr_loss" % (
        opt.lr,
        opt.epochs,
        opt.batch_size,
        opt.attention_window[0],
        opt.backbone_lr_factor,
        str(opt.pretrained),
    )
    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
    )
    opt.sfx = str(
        "%s.unint_act.%s.layers%d.attn_win%d.time%s_cntr_loss"
        % (
            opt.dataset,
            opt.task,
            opt.num_hidden_layers,
            opt.attention_window[0],
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )

    setup_logger_path()

    if opt.use_crf:
        opt.balance_fails_only = False
        opt.all_fail_videos = False
        opt.load_videos = True
        opt.step_between_clips_sec = 0.25
        train_loader = get_video_loader_trn_2x(opt)
        opt.val = True
        opt.fails_path = (
            "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
        )
        val_loader = get_video_loader_trn_2x(opt)
    else:
        opt.balance_fails_only = True
        opt.all_fail_videos = False
        opt.load_videos = False
        if opt.backbone == "vit_longformer":
            train_loader = get_video_loader(opt)
        else:
            train_loader = get_video_loader_frames(opt)
        opt.val = True
        opt.fails_path = (
            "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
        )
        if opt.backbone == "vit_longformer":
            val_loader = get_video_loader(opt)
        else:
            val_loader = get_video_loader_frames(opt)

    if opt.backbone == "vit_longformer":
        model, optimizer, loss = create_model(
            num_classes=100 if opt.dataset == "kinetics" else opt.num_classes,
            pretrained=opt.pretrained,
        )
    elif opt.backbone == "r3d_18":
        model, optimizer, loss = create_r3d(opt.pretrained)

    epoch = 0

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epochs=25,
        epoch=epoch,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    do_uar()
