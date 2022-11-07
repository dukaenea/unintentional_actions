# @Author: Enea Duka
# @Date: 5/7/21

import torch
import os
import copy
from tqdm import tqdm
from datetime import datetime
from rep_learning.test import test
from action_classification.test import test as test_classify
import torch.nn.functional as F
from utils.util_functions import label_idx_to_one_hot, adjust_lr, AIAYNScheduler
from utils.model_saver import ModelSaver
from utils.util_functions import Meter
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.util_functions import lr_func_cosine, contrastive_loss
from utils.util_functions import Precision, label_idx_to_one_hot
from utils.util_functions import DistributionPlotter
from transformer2x.trn_utils import prep_for_local, prep_for_global


def train(**kwargs):
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    val_loader = kwargs["val_loader"]
    optimizer = kwargs["optimizer"]
    loss = kwargs["loss"]
    tst = datetime.now().strftime("%Y%m%d-%H%M%S")
    s_epoch = kwargs["epoch"]

    loss_meter = Meter("train")
    bce_meter = Meter(mode="train", name="bce_loss")
    model_saver = ModelSaver(
        path=os.path.join(
            opt.storage,
            "models",
            opt.dataset,
            opt.model_name,
            opt.sfx,
            opt.log_name,
            "val",
        )
    )
    if opt.lr_scheduler == "aiayn":  # Attention is All You Need
        aiayn_scheduler = AIAYNScheduler(
            opt.hidden_dim, 0.1 * (len(train_loader) * opt.epochs)
        )  # 10% of the steps for the warmup
    test(model=model, loss=loss, dataloader=val_loader, mode="val", time=tst, epoch=-1)

    logger.debug("Starting training for %d epochs:" % opt.epochs)
    prec = Precision("train")
    new_lr = opt.lr

    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1 / 8]).cuda())

    if opt.task == "regression":
        out_dist_plotter = DistributionPlotter()

    lrs = [1e-4, 8e-5, 5e-5, 3e-5, 1e-5]
    curr_lr = opt.lr

    for epoch in range(s_epoch, opt.epochs):
        model.train()
        new_lr = lr_func_cosine(
            opt.lr, opt.lr * opt.cos_decay_lr_factor, opt.epochs, epoch
        )

        logger.debug("New LR: %f" % new_lr)
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = new_lr * opt.backbone_lr_factor

        if opt.use_tqdm:
            iterator = enumerate(tqdm(train_loader))
        else:
            iterator = enumerate(train_loader)

        for idx, data in iterator:
            pure_nr_frames = data["pure_nr_frames"]
            labels = data["label"]
            keys = list(data.keys())
            videos = data.get(keys[0])
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

            loss_meter.update(_loss.item(), videos.shape[0])
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if idx % 100 == 0 and idx > 0:
                logger.debug("Loss: %f" % loss_meter.avg)

        logger.debug("Loss: %f" % loss_meter.avg)
        logger.debug("BCE Loss: %f" % loss_meter.avg)
        loss_meter.reset()
        bce_meter.reset()
        check_val = None
        # logger.debug('Epoch %d ==== Speed&Motoion TrainAcc: %f' % (epoch, prec.top1()))
        if opt.task == "regression":
            out_dist_plotter.plot_out_dist()
        if (opt.test_freq and epoch % opt.test_freq == 0) or epoch == (opt.epochs - 1):
            # opt.rep_learning = False
            # opt.viz = False
            check_val = test(
                model=model,
                loss=loss,
                dataloader=val_loader,
                mode="val",
                time=tst,
                epoch=epoch,
            )

        if not opt.use_tqdm:
            print("=====================================================")

        if (
            (opt.save_model and epoch % opt.save_model == 0)
            or epoch == (opt.epochs - 1)
        ) and check_val:
            if model_saver.check(check_val):
                save_dict = {
                    "epoch": epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict().copy()),
                }
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()

    return model
