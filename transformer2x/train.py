# @Author: Enea Duka
# @Date: 5/7/21

import torch
import os
import copy
from tqdm import tqdm
from datetime import datetime
from transformer2x.test import test
import torch.nn.functional as F
from utils.util_functions import AIAYNScheduler
from utils.model_saver import ModelSaver
from utils.util_functions import Meter
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.util_functions import lr_func_cosine
from utils.util_functions import Precision
from transformer2x.trn_utils import prep_for_local, prep_for_global, prep_for_crf
import math
from transformer2x.test_localisation import test_localisation


def train(**kwargs):
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    val_loader = kwargs["val_loader"]
    optimizer = kwargs["optimizer"]
    loss = kwargs["loss"]
    tst = datetime.now().strftime("%Y%m%d-%H%M%S")

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
    model.eval()
    test(
        model=model, loss=loss, dataloader=val_loader, mode="val", time=tst, epoch=-1,
    )
    logger.debug(str(loss.weight))
    logger.debug("Starting training for %d epochs:" % opt.epochs)

    zeros = 0
    ones = 0
    twos = 0

    for epoch in range(opt.epochs):
        model.train()

        new_lr = lr_func_cosine(
            opt.lr, opt.lr * opt.cos_decay_lr_factor, opt.epochs, epoch
        )
        logger.debug("New LR: %f" % new_lr)

        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        if opt.use_tqdm:
            iterator = enumerate(tqdm(train_loader))
        else:
            iterator = enumerate(train_loader)

        for idx, data in iterator:
            optimizer.zero_grad()
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
                if videos.shape[0] > 32:
                    chunks = torch.split(videos, 32)
                    v = []
                    for chunk in chunks:
                        chunk_len = -1
                        if chunk.shape[0] < 4:
                            chunk_len = chunk.shape[0]
                            remaining = 4 - chunk.shape[0]
                            pad = torch.zeros(
                                (
                                    remaining,
                                    chunk.shape[1],
                                    chunk.shape[2],
                                    chunk.shape[3],
                                    chunk.shape[4],
                                )
                            ).to(chunk.device)
                            chunk = torch.cat((chunk, pad))
                        out = model(
                            chunk,
                            position_ids,
                            None,
                            pnf,
                            labels,
                            local=True,
                            multi_scale=opt.multi_scale,
                        )
                        if chunk_len != -1:
                            out = out[:chunk_len]
                        v.append(out)
                    videos = torch.cat(v, dim=0)
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
                    videos, position_ids, None, pure_nr_frames, labels, num_clips, False
                )
            else:
                num_clips = pure_nr_frames.shape[0]
                out = videos

            if opt.use_crf:
                videos = prep_for_crf(out, pure_nr_frames)
                if opt.crf_margin_probs:
                    out = model(
                        videos,
                        position_ids,
                        None,
                        pure_nr_frames,
                        labels,
                        num_clips,
                        False,
                        True,
                    )
                    out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])
                    labels = labels.cuda().flatten()
                    mask = labels != -1
                    labels = labels[mask]
                    out = out[mask]
                    _loss = loss(out, labels)
                else:
                    _loss = model(
                        videos, None, None, pure_nr_frames, labels, None, False, True
                    ).mean()
            else:
                _loss = loss(out, labels.cuda())

            loss_meter.update(_loss.item(), videos.shape[0])
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if idx % 100 == 0:
                logger.debug("Train Loss: %f" % loss_meter.avg)

        print(zeros)
        print(ones)
        print(twos)
        logger.debug("Train Loss: %f" % loss_meter.avg)
        loss_meter.reset()
        # logger.debug('Train Acc: %f' % prec.top1())
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
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

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {
                    "epoch": epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict().copy()),
                }
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()
