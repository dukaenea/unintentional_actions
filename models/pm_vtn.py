# code from https://github.com/bomri/SlowFast/blob/master/slowfast
# @Author: Enea Duka
# @Date: 5/3/21
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_setup import logger
from torch.autograd import Function

from torchcrf import CRF
from itertools import chain
from utils.arg_parse import opt


class VTNLongformerModel(transformers.LongformerModel):
    def __init__(
        self,
        embed_dim=768,
        max_positions_embedding=2 * 60 * 60,
        num_attention_heads=12,
        num_hidden_layers=3,
        attention_mode="sliding_chunks",
        pad_token_id=-1,
        attention_window=None,
        intermediate_size=3072,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    ):
        self.config = transformers.LongformerConfig()
        self.config.attention_mode = attention_mode
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_dilation = [1,] * num_attention_heads
        self.config.attention_window = (
            [256,] * num_hidden_layers if attention_window is None else attention_window
        )
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_positions_embedding
        self.config.hidden_size = embed_dim
        super(VTNLongformerModel, self).__init__(self.config, add_pooling_layer=False)
        self.embeddings.word_embeddings = None


def pad_to_window_size_local(
    input_ids,
    attention_mask,
    position_ids,
    one_sided_window_size,
    pad_token_id,
    pure_nr_frames=None,
):
    w = 2 * one_sided_window_size
    seq_len = input_ids.size(1)
    padding_len = (w - seq_len % w) % w
    input_ids = F.pad(
        input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id
    ).permute(0, 2, 1)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)
    position_ids = F.pad(position_ids, (1, padding_len), value=False)

    if pure_nr_frames is not None:
        for idx, nr_frames in enumerate(pure_nr_frames):
            try:
                attention_mask[idx, nr_frames:] = 0
                position_ids[idx, nr_frames:] = 1
            except Exception as e:
                pass
    return input_ids, attention_mask, position_ids


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self._build_model()

    def _build_model(self):
        # self.backbone = vit_base_patch16_224(pretrained=True,
        #                                      num_classes=0,
        #                                      drop_path_rate=0,
        #                                      drop_rate=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, opt.embed_dim))
        self.lin_projector = nn.Linear(3 * opt.cuboid_resolution ** 3, opt.embed_dim)
        if opt.use_crf:
            self.crf = CRF(num_tags=3, batch_first=True)

        self.temporal_encoder = VTNLongformerModel(
            embed_dim=opt.embed_dim,
            max_positions_embedding=opt.max_positions_embedding,
            num_attention_heads=opt.num_attention_heads,
            num_hidden_layers=opt.num_hidden_layers,
            attention_mode=opt.attention_mode,
            pad_token_id=opt.pad_token_id,
            attention_window=opt.attention_window,
            intermediate_size=opt.intermediate_size,
            attention_probs_dropout_prob=opt.attention_probs_dropout_prob,
            hidden_dropout_prob=opt.hidden_dropout_prob,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            nn.BatchNorm1d(opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(
                opt.mlp_dim,
                opt.num_classes_ptr
                if opt.num_classes_ptr is not None
                else opt.num_classes,
            ),
        )

    def forward(
        self,
        x,
        position_ids,
        speed_labels=None,
        pure_nr_frames=None,
        return_features=False,
        classifier_only=False,
        backbone_only=False,
        backbone_feats_only=False,
        high_order_temporal_features=False,
        for_crf=False,
        labels=None,
    ):
        if classifier_only:
            x = self.mlp_head(x)
            return x

        if len(x.shape) == 4:
            B, CL, F, C = x.shape
            x = x.reshape(B * CL, F, C)
            position_ids = position_ids.reshape(B * CL, F)
            pure_nr_frames = pure_nr_frames.flatten()

        if len(x.shape) == 5:
            B, F, C, H, W = x.shape
            x = x.reshape(B * F, C, H, W)
            x = self.backbone(x)
            x = x.reshape(B, F, -1)
        if for_crf:
            # create the mask for the CRF layer
            crf_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.uint8).to(
                x.device
            )
            mask = (x == 0)[:, :, 0]
            crf_mask[mask] = 0
            if self.training:
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(0)
                loss = -self.crf.forward(x, labels, mask=crf_mask, reduction="mean")
                return loss
            else:
                loss = -self.crf.forward(x, labels, mask=crf_mask, reduction="mean")
                mls = self.crf.decode(x, mask=crf_mask)
                return torch.tensor(list(chain.from_iterable(mls))).to(x.device), loss

        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        cls_attn = torch.ones((1)).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_attn), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id,
            pure_nr_frames,
        )
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=x,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )

        x = x["last_hidden_state"]
        x = x[:, 0]

        if return_features:
            return x
        x_mlp = self.mlp_head(x)
        return x_mlp

    def reinit_mlp(self, num_classes):
        if opt.task == "classification":
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(opt.hidden_dim),
                nn.Linear(opt.hidden_dim, opt.mlp_dim),
                # nn.BatchNorm1d(opt.mlp_dim),
                nn.GELU(),
                nn.Dropout(opt.mlp_dropout),
                nn.Linear(opt.mlp_dim, num_classes),
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(opt.hidden_dim),
                nn.Linear(opt.hidden_dim, opt.mlp_dim),
                nn.GELU(),
                nn.Dropout(opt.mlp_dropout),
                nn.Linear(opt.mlp_dim, 1),
                nn.Sigmoid(),
            )
        self.mlp_head.cuda()

    def freeze(self):
        for name, child in self.named_children():
            if name == "temporal_encoder":
                for param in child.parameters():
                    param.requires_grad = False


def get_froze_trn_optimizer(model):
    if opt.optim == "adam":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.module.mlp_head.parameters(), "lr": opt.lr},
                {
                    "params": model.module.temporal_encoder.parameters(),
                    "lr": opt.lr * opt.backbone_lr_factor,
                },
            ],
            weight_decay=opt.weight_decay,
        )
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": model.module.mlp_head.parameters(), "lr": opt.lr},
                {"params": model.module.temporal_encoder.parameters(), "lr": 0},
            ],
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )

    return optimizer


def create_model(num_classes=None, pretrained=False):
    model = VTN()
    if num_classes is not None and not pretrained:
        model.reinit_mlp(num_classes)
    model.cuda()
    if opt.gpu_parallel:
        model = nn.DataParallel(model)
        if pretrained:
            saved_model = torch.load(opt.vtn_ptr_path)
            try:
                model_dict = saved_model["state_dict"]
            except KeyError:
                model_dict = saved_model["vtn_state_dict"]
            model.load_state_dict(model_dict, strict=False)
            if opt.gpu_parallel:
                # model.module.freeze()
                if num_classes is not None:
                    model.module.reinit_mlp(num_classes)
                model.module.cuda()
                if opt.gpu_parallel:
                    model = nn.DataParallel(model.module)
            else:
                model.freeze()
                model.reinit_mlp(num_classes)

    # part_freeze_vit(model)
    # freeze_model(model)
    if opt.optim == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), opt.lr, weight_decay=opt.weight_decay
        )
    # optimizer = None
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, momentum=opt.momentum
        )

    if opt.task == "classification":
        loss = nn.CrossEntropyLoss()
    elif opt.task == "regression":
        loss = nn.SmoothL1Loss(beta=0.01, reduction="sum")

    logger.debug(str(model))
    logger.debug(str(optimizer))
    logger.debug(str(loss))
    return model, optimizer, loss


def part_freeze_vit(model):
    for param in model.named_parameters():
        if "module.backbone" in param[0]:
            if (
                ("module.backbone.blocks.9" in param[0])
                or ("module.backbone.blocks.10" in param[0])
                or ("module.backbone.blocks.11" in param[0])
                or ("module.backbone.norm" in param[0])
            ):
                print(param[0])
            else:
                param[1].requires_grad = False
                print(param[0])


def freeze_model(model):
    frozen_param_names = []
    for param in model.named_parameters():
        if (
            "module.mlp_head" not in param[0]
            and "module.bc_head" not in param[0]
            and "module.high_order_temporal_encoder" not in param[0]
        ):
            param[1].requires_grad = False
            frozen_param_names.append(param[0])

    return frozen_param_names


def unfreeze_model(model, frozen_param_names):
    for param in model.named_parameters():
        if param[0] in frozen_param_names:
            param[1].requires_grad = True
