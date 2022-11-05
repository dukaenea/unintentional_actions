# @Author: Enea Duka
# @Date: 5/7/21

import torch
import os
import copy
from tqdm import tqdm
from datetime import datetime
from action_localization.test_regression import test
from utils.model_saver import ModelSaver
from utils.util_functions import Meter
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.util_functions import lr_func_cosine
from utils.util_functions import Precision


def train(**kwargs):
    model = kwargs["model"]
    feat_extractor = kwargs["feat_extractor"]
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
    feat_extractor.eval()
    test(
        model=model,
        feat_extractor=feat_extractor,
        loss=loss,
        dataloader=val_loader,
        mode="val",
        time=tst,
        epoch=-1,
    )

    logger.debug("Starting training for %d epochs:" % opt.epochs)
    new_lr = opt.lr
    prec = Precision("train")
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
            position_ids = (
                torch.tensor(list(range(0, videos.shape[1])))
                .expand(1, videos.shape[1])
                .repeat(videos.shape[0], 1)
            )
            out = model(videos, position_ids, None, pure_nr_frames)
            if opt.task == "classification":
                labels = data["label"]
                _loss = loss(out, labels.cuda())
            elif opt.task == "regression":
                out = out.permute(1, 0).squeeze()
                labels = data["rel_t"].type(torch.float32)
                lengths = data["len"]
                _loss = loss(out, labels.cuda())

            loss_meter.update(_loss.item(), videos.shape[0])
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        logger.debug("Train Loss: %f" % loss_meter.avg)
        loss_meter.reset()
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(
                model=model,
                feat_extractor=feat_extractor,
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
