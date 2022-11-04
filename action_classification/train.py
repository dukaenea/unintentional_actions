# @Author: Enea Duka
# @Date: 5/7/21

import torch
import os
import copy
from tqdm import tqdm
from datetime import datetime
from action_classification.test import test
from utils.util_functions import label_idx_to_one_hot, adjust_lr, AIAYNScheduler
from utils.model_saver import ModelSaver
from utils.util_functions import Meter
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.util_functions import lr_func_cosine
from utils.util_functions import Precision
from transformer2x.trn_utils import prep_for_local, prep_for_crf
import math

from models.pm_vtn import freeze_model, unfreeze_model


def train(**kwargs):
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    val_loader = kwargs["val_loader"]
    optimizer = kwargs["optimizer"]
    loss = kwargs["loss"]
    tst = datetime.now().strftime("%Y%m%d-%H%M%S")
    s_epoch = kwargs["epoch"]

    loss_meter = Meter("train")
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
    # if opt.lr_scheduler == 'aiayn': # Attention is All You Need
    aiayn_scheduler = AIAYNScheduler(
        opt.hidden_dim, 0.3 * (len(train_loader) * opt.epochs)
    )  # 10% of the steps for the warmup

    test(model=model, loss=loss, dataloader=val_loader, mode="val", time=tst, epoch=-1)
    logger.debug(str(loss.weight))

    logger.debug("Starting training for %d epochs:" % opt.epochs)
    new_lr = opt.lr
    prec = Precision("train")
    # frozen_param_names = freeze_model(model)

    for epoch in range(s_epoch, opt.epochs):
        model.train()

        # unfreeze_model(model, frozen_param_names)
        new_lr = lr_func_cosine(
            opt.lr, opt.lr * opt.cos_decay_lr_factor, opt.epochs, epoch
        )
        logger.debug("New LR: %f" % new_lr)
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = new_lr * opt.backbone_lr_factor

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lr

        if opt.use_tqdm:
            iterator = enumerate(tqdm(train_loader))
        else:
            iterator = enumerate(train_loader)

        for idx, data in iterator:
            optimizer.zero_grad()
            videos = data["features"]
            pure_nr_frames = data["pure_nr_frames"]
            labels = data["label"]

            if opt.use_crf:
                videos, position_ids, pnf = prep_for_local(videos, pure_nr_frames)
                out = model(videos, position_ids, None, pure_nr_frames)
                pure_nr_frames = torch.t(pure_nr_frames)[0]
                videos = prep_for_crf(out, pure_nr_frames)
                _loss = model(
                    videos, None, None, pure_nr_frames, labels=labels, for_crf=True
                ).mean()
            else:
                if opt.backbone == "vit_longformer":
                    position_ids = (
                        torch.tensor(list(range(0, videos.shape[1])))
                        .expand(1, videos.shape[1])
                        .repeat(videos.shape[0], 1)
                    )
                    # videos = videos.mean(1).squeeze()
                    out = model(
                        videos,
                        position_ids,
                        None,
                        pure_nr_frames,
                        classifier_only=False,
                    )
                else:
                    out = model(videos)

                # next_frames = data['next_frame']
                # try:
                _loss = loss(out, labels.cuda())
            # except Exception as e:
            # pass
            # else:
            # _loss = mmargin_contrastive_loss(out_c, labels.cuda())
            # ce_loss = loss(out, labels.cuda())
            # _loss = 10 * _loss + ce_loss
            #     # _loss = loss(out, labels.cuda())

            loss_meter.update(_loss.item(), videos.shape[0])
            # prec.update_probs_reg(out, labels.cuda(), lengths.cuda())
            # a = model.module.temporal_encoder.encoder.layer[0].output.dense.weight
            # logger.debug(str(a))
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        logger.debug("Train Loss: %f" % loss_meter.avg)
        loss_meter.reset()
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(
                model=model,
                loss=loss,
                dataloader=val_loader,
                time=tst,
                mode="val",
                epoch=epoch,
            )
        if not opt.use_tqdm:
            print("=====================================================")

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {
                    "epoch": epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict().copy()),
                }
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()
