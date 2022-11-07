# @Author: Enea Duka
# @Date: 5/3/21
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.oops_ttibua_loader import OopsTtibua
from torch.utils.data import DataLoader

from models.pm_vtn import create_model
from models.res3d_18 import create_r3d
from transformer2x.vtn import create_model_trn_2x
from rep_learning.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime
import warnings


def learn_representation():

    opt.sfx = str(
        "%s.rep_learning.tag:%s.layers%d.attn_win%d.classes%d_rok_class_per_trn.task_%s"
        % (
            opt.dataset,
            opt.tag,
            opt.num_hidden_layers,
            opt.attention_window[0],
            opt.num_classes,
            opt.task,
        )
    )

    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
    )
    opt.sfx = str(
        "%s.rep_learning.tag:%s.layers%d.attn_win%d.classes%d.time%s"
        % (
            opt.dataset,
            opt.tag,
            opt.num_hidden_layers,
            opt.attention_window[0],
            opt.num_classes,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )
    if opt.debug:
        opt.num_workers = 0
        opt.batch_size = 5
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
    elif opt.dataset == "rareact":
        pass
    elif opt.dataset == "oops":
        pass
    elif opt.dataset == "all":
        train_set = OopsTtibua(
            "train",
            spat_scale=True,
            size=224,
            spat_crop=True,
            load_frames=True if opt.backbone == "r3d_18" else False,
        )
        val_set = OopsTtibua(
            "val",
            spat_scale=True,
            size=224,
            spat_crop=True,
            load_frames=True if opt.backbone == "r3d_18" else False,
        )

    if opt.dataset == "all":
        train_loader = DataLoader(
            train_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_set.speed_and_motion_collate_fn
            if not opt.multi_scale
            else train_set.video_level_speed_and_motion_collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=val_set.speed_and_motion_collate_fn
            if not opt.multi_scale
            else train_set.video_level_speed_and_motion_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False if opt.debug else True,
            drop_last=True,
            collate_fn=train_set._rep_lrn_collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False if opt.debug else True,
            drop_last=True,
            collate_fn=val_set._rep_lrn_collate_fn,
        )

    if opt.multi_scale:
        model, optimizer, loss = create_model_trn_2x(
            opt.num_classes, pretrained=opt.pretrained, pretrain_scale="frame"
        )
    else:
        if opt.backbone == "vit_longformer":
            model, optimizer, loss = create_model(
                opt.num_classes, pretrained=opt.pretrained
            )
        elif opt.backbone == "r3d_18":
            model, optimizer, loss = create_r3d(pretrained=opt.pretrained)

    epoch = 0

    # if opt.pretrained:
    #     saved_model = torch.load(opt.vtn_ptr_path)
    #     optimizer.load_state_dict(saved_model['optimizer'])
    #     epoch = saved_model['epoch'] + 1

    # opt.batch_size = 256
    # opt.workers = 32
    # opt.balance_fails_only = True
    # opt.all_fail_videos = False
    # # train_loader = get_video_loader(opt)
    # opt.val = True
    # opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
    # val_loader_class = get_video_loader(opt)
    # opt.batch_size = 32

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epochs=25,
        train_set=train_set,
        epoch=epoch,
    )

    return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    learn_representation()
