# @Author: Enea Duka
# @Date: 5/3/21
import sys

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")
import os
from dataloaders.my_oops_loader import get_video_loader
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
from models.pm_vtn import create_model
from models.resnet50 import build_resnet_50
from action_localization.train_regression import train
from action_localization.test_localization import test
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime
from torch.utils.data.dataloader import default_collate


def main():
    opt.dataset = "oops"
    opt.rep_data_level = "features"
    opt.backbone = "resnet"
    opt.rep_backbone = "resnet"
    opt.lr_scheduler = "step"
    opt.embed_dim = 768
    opt.intermediate_size = 3072
    opt.hidden_dim = 768
    opt.num_classes = 3
    opt.num_classes_ptr = 3

    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001_v0.5402_ep48.pth.tar"  # best one so far 62.81% val acc
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32_best_classification/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32_best_classification_v0.6251_ep32.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210706-230147/lr:0.000100~ep:50~bs:256~win:32~b_lr:1.000000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210706-230147_v0.6185_ep47.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-004035_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-004035_cntr_loss_v0.6418_ep31.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210828-214755_cntr_loss/lr:0.000100~ep:20~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210828-214755_cntr_loss_v0.6525_ep0.pth.tar"

    opt.model_name = "VideoLongformer_MLP"
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 1 if opt.task == "classification" else 16
    opt.sfx = str(
        "%s.unint_act_loc.%s.layers%d.attn_win%d.task%s"
        % (
            opt.dataset,
            opt.task,
            opt.num_hidden_layers,
            opt.attention_window[0],
            opt.task,
        )
    )
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False

    opt.optim = "adam"
    opt.momentum = 0.9
    opt.lr = 1e-5
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 1e-2
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1
    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
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
    if opt.dataset == "oops":
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
            loc_class=True if opt.task == "classification" else False,
        )

    if opt.dataset == "oops" and opt.task == "classification":
        opt.batch_size = 128
        opt.sample_videos = True
        opt.workers = 32
        opt.balance_fails_only = True
        opt.all_fail_videos = False
        train_loader = get_video_loader(opt)
        opt.val = True
        opt.fails_path = (
            "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
        )
        val_loader = get_video_loader(opt)

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
            shuffle=True,
            drop_last=True,
            collate_fn=train_set.pad_videos_collate_fn
            if opt.task == "regression"
            else default_collate,
        )

    model, optimizer, loss = create_model(
        pretrained=True, num_classes=1 if opt.task == "regression" else None
    )
    feat_extractor = build_resnet_50(None)
    if opt.task == "classification":
        test(
            model=model,
            loss=loss,
            dataloader=val_loader,
            mode="val",
            epoch=-1,
            time=datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    else:
        train(
            model=model,
            feat_extractor=feat_extractor,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss=loss,
            test_freq=1,
            epochs=25,
        )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
