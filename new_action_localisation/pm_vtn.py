# code from https://github.com/bomri/SlowFast/blob/master/slowfast
# @Author: Enea Duka
# @Date: 5/3/21
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_setup import logger
from torch.autograd import Function
from timm.models.vision_transformer import vit_base_patch16_224

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


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self._build_model()

    def _build_model(self):
        self.backbone = vit_base_patch16_224(
            pretrained=True, num_classes=0, drop_path_rate=0, drop_rate=0
        )
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

        # self.high_order_temporal_encoder = VTNLongformerModel(
        #     embed_dim=opt.embed_dim,
        #     max_positions_embedding=opt.max_positions_embedding,
        #     num_attention_heads=opt.num_attention_heads,
        #     num_hidden_layers=1,
        #     attention_mode=opt.attention_mode,
        #     pad_token_id=opt.pad_token_id,
        #     attention_window=[8],
        #     intermediate_size=opt.intermediate_size,
        #     attention_probs_dropout_prob=opt.attention_probs_dropout_prob,
        #     hidden_dropout_prob=opt.hidden_dropout_prob
        # )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            # nn.BatchNorm1d(opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(
                opt.mlp_dim,
                opt.num_classes_ptr
                if opt.num_classes_ptr is not None
                else opt.num_classes,
            ),
        )

        self.loc_mlp = nn.Sequential(
            nn.LayerNorm(opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            # nn.BatchNorm1d(opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(opt.mlp_dim, 17),
        )

        # self.bc_head = nn.Sequential(
        #     nn.LayerNorm(opt.hidden_dim),
        #     nn.Linear(opt.hidden_dim, opt.mlp_dim),
        #     # nn.BatchNorm1d(opt.mlp_dim),
        #     nn.GELU(),
        #     nn.Dropout(opt.mlp_dropout),
        #     nn.Linear(opt.mlp_dim, 1),
        #     nn.Sigmoid()
        # )

        # self.mem_head = nn.Linear(opt.hidden_dim*2, opt.hidden_dim)
        # self.frame_mem_head = nn.Linear(opt.hidden_dim*2, opt.hidden_dim)
        # self.tanh = nn.Tanh()
        # self.domain_classifier = nn.Sequential(
        #     nn.LayerNorm(opt.hidden_dim),
        #     nn.Linear(opt.hidden_dim, opt.mlp_dim),
        #     nn.GELU(),
        #     nn.Dropout(opt.mlp_dropout),
        #     nn.Linear(opt.mlp_dim, opt.num_classes)
        # )
        #
        # self.memory = IncrementalParametricMemory(1024, opt.embed_dim)
        # self.frame_memory = IncrementalParametricMemory(1024, opt.embed_dim)

    def forward(
        self,
        x,
        position_ids=None,
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
        if position_ids is None:
            return self.loc_mlp(x)

        if classifier_only:
            x = self.mlp_head(x)
            return x
        #
        # ORG_BATCH = None
        if len(x.shape) == 4:
            B, CL, F, C = x.shape
            ORG_BATCH = B
            x = x.reshape(B * CL, F, C)
            position_ids = position_ids.reshape(B * CL, F)
            pure_nr_frames = pure_nr_frames.flatten()

        if len(x.shape) == 5:
            B, F, C, H, W = x.shape
            # x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(B * F, C, H, W)
            x = self.backbone(x)
            # mem_values = self.frame_memory(x)
            # x = torch.cat((x, mem_values), dim=1)
            # x = self.frame_mem_head(x)
            x = x.reshape(B, F, -1)
        #
        # if backbone_feats_only:
        #     return x
        #
        # if backbone_only:
        #     x = self.mlp_head(x)
        #     return x
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

        # if high_order_temporal_features:
        #     if ORG_BATCH is not None:
        #         seq_len = x.shape[0] // ORG_BATCH
        #         x = torch.stack(list(x.split(seq_len)), dim=0)
        #     # x = x_bc.unsqueeze(0)
        #     # B = 1
        #     B, D, E = x.shape
        #     attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        #     # cls_token = self.hl_cls_token.expand(B, -1, -1)
        #     # x = torch.cat((cls_token, x), dim=1)
        #     # cls_attn = torch.ones((1)).expand(B, -1).to(x.device)
        #     # attention_mask = torch.cat((attention_mask, cls_attn), dim=1)
        #     # attention_mask[:, 0] = 2
        #     if ORG_BATCH is not None:
        #         position_ids = torch.arange(0, D).repeat(ORG_BATCH, 1).to(x.device)
        #
        #     x, attention_mask, position_ids = pad_to_window_size_local(
        #         x,
        #         attention_mask,
        #         position_ids,
        #         self.high_order_temporal_encoder.config.attention_window[0],
        #         self.high_order_temporal_encoder.config.pad_token_id
        #     )
        #     token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        #     # token_type_ids[:, 0] = 1
        #
        #     mask = attention_mask.ne(0).int()
        #     max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        #     position_ids = position_ids % (max_position_embeddings - 2)
        #     position_ids[:, 0] = max_position_embeddings - 2
        #     position_ids = position_ids[:, 1:]
        #     position_ids[mask == 0] = max_position_embeddings - 1
        #
        #     x = self.high_order_temporal_encoder(input_ids=None,
        #                                           attention_mask=attention_mask,
        #                                           token_type_ids=token_type_ids,
        #                                           position_ids=position_ids,
        #                                           inputs_embeds=x,
        #                                           output_attentions=None,
        #                                           output_hidden_states=True,
        #                                           return_dict=True)
        #
        #     x = x['last_hidden_state']
        #     x = x[:, :pure_nr_frames[0]]
        #     # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        #     # x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        #

        # at this point we have the embeddings of the clips
        # we use the embeddings as queries for the memory
        # mem_values = self.memory(x)
        # x = torch.cat((x, mem_values), dim=1)
        # x = self.mem_head(x)
        # x = F.sigmoid(x)
        if return_features:
            # x = torch.tanh(x)
            # x = F.relu(x)
            # x = F.sigmoid(x)
            return x
        x_mlp = self.mlp_head(x)
        # x_bc = self.bc_head(x)
        # x_mlp_dom = self.grad_reverse(x)
        # x_mlp_dom = self.domain_classifier(x_mlp_dom)
        # x_bc = torch.stack(list(torch.split(x_bc, pure_nr_frames[0])), dim=0).squeeze()
        return x_mlp, x

    def grad_reverse(self, x):
        return GradientReversal.apply(x)

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
            # self.bc_head = nn.Sequential(
            #     nn.LayerNorm(opt.hidden_dim),
            #     nn.Linear(opt.hidden_dim, opt.mlp_dim),
            #     # nn.BatchNorm1d(opt.mlp_dim),
            #     nn.GELU(),
            #     nn.Dropout(opt.mlp_dropout),
            #     nn.Linear(opt.mlp_dim, 1),
            #     nn.Sigmoid()
            # )
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

    def expand_memory(self, expand_size):
        self.memory.expand_memory(expand_size)


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

        # if opt.rep_learning:
        # optimizer = torch.optim.AdamW([{'params': model.module.cls_token, 'lr': opt.lr * 0},
        #                                {'params': model.module.temporal_encoder.parameters(), 'lr': opt.lr * 0},
        #                                # {'params': model.module.high_order_temporal_encoder.parameters(), 'lr': opt.lr},
        #                                {'params': model.module.backbone.parameters(), 'lr': opt.lr * 0},
        #                                {'params': model.module.mlp_head.parameters(), 'lr': opt.lr},
        #                                {'params': model.module.crf.parameters(), 'lr': opt.lr}
        #                                # {'params': model.module.bc_head.parameters(), 'lr': opt.lr}],
        #                                ],
        #                               weight_decay=opt.weight_decay)
        # else:
        optimizer = torch.optim.AdamW(
            model.parameters(), opt.lr, weight_decay=opt.weight_decay
        )
    # optimizer = None
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, momentum=opt.momentum
        )

    # if opt.dataset == 'kinetics':
    if opt.task == "classification":
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.25, 0.4, 0.25]).cuda())ss
        loss = nn.CrossEntropyLoss()
        # loss = nn.BCELoss()
        # loss = nn.MSELoss(reduction='sum')
    elif opt.task == "regression":
        # loss = nn.MSELoss(reduction='sum')
        loss = nn.SmoothL1Loss(beta=0.01, reduction="sum")
        # loss = RegressionLoss()

    # elif opt.dataset == 'oops':
    #     loss = nn.BCEWithLogitsLoss()

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


class MILRegLoss(nn.Module):
    def __init__(self, model, lambdas=0.001):
        super(MILRegLoss, self).__init__()

        self.lambdas = lambdas
        self.model = model

    def forward(self, y_pred, y_true):

        # fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.module.bc_head.layer[1].parameters()]))
        # fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.module.bc_head.layer[4].parameters()]))

        loss_reg = 0
        # for param in self.model.parameters():
        #     print(param)

        # l1_reg = self.lambdas * torch.norm(fc1_params, p=2)
        # l2_reg = self.lambdas * torch.norm(fc2_params, p=2)
        #
        return self.mil_objective(y_pred, y_true)

    def mil_objective(self, y_pred, y_true):
        labmdas = 8e-5
        y_true = y_true.reshape(y_true.shape[0] * y_true.shape[1])
        normal_labels_idx = y_true == 0
        anom_labels_idx = y_true == 1

        y_pred = y_pred.squeeze()
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])
        normal_vid_sc = y_pred[normal_labels_idx].squeeze()
        anomal_vid_sc = y_pred[anom_labels_idx].squeeze()

        # normal_vid_sc = y_pred[:batch_size].squeeze()
        # anomal_vid_sc = y_pred[batch_size:].squeeze()

        # normal_vid_sc = torch.stack(list(torch.split(normal_vid_sc, 10))).mean(1).squeeze()
        # anomal_vid_sc = torch.stack(list(torch.split(anomal_vid_sc, 10))).mean(1).squeeze()

        # normal_segs_max_scores = normal_vid_sc.mean(dim=-1)
        # anomal_segs_max_scores = anomal_vid_sc.mean(dim=-1)
        # hinge_loss = 1 - torch.max(torch.zeros_like(anomal_segs_max_scores), anomal_segs_max_scores - normal_segs_max_scores)

        normal_segs_max_scores = normal_vid_sc.max(dim=-1)[0]
        normal_segs_min_scores = normal_vid_sc.min(dim=-1)[0]

        # anomal_segs_max_scores = anomal_vid_sc.max(dim=-1)[0]
        anomal_segs_min_scores = anomal_vid_sc.min(dim=-1)[0]

        # normal_segs_max_scores = torch.topk(normal_vid_sc, 3, dim=1)[0].mean(1)
        anomal_segs_max_scores = torch.topk(anomal_vid_sc, 1, dim=1)[0].mean(1)
        # anomal_segs_min_scores = torch.topk(anomal_vid_sc, k=(32-3), dim=1, largest=False)[0].mean(1)

        hinge_loss = 1 - anomal_segs_max_scores + normal_segs_max_scores
        hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

        hinge_loss_ab = 1 - anomal_segs_max_scores + anomal_segs_min_scores
        hinge_loss_ab = torch.max(hinge_loss_ab, torch.zeros_like(hinge_loss_ab))

        loss_nb = torch.abs(normal_segs_max_scores - normal_segs_min_scores)

        smoothed_scores = anomal_vid_sc[:, 1:] - anomal_vid_sc[:, :-1]
        smoothed_scores_ss = smoothed_scores.pow(2).sum(dim=-1)

        sparsity_loss = anomal_vid_sc.sum(dim=-1)
        # an_sp_loss = (anomal_segs_max_scores - anomal_segs_min_scores).mean()

        final_loss = (
            hinge_loss + labmdas * smoothed_scores_ss + labmdas * sparsity_loss
        ).mean()

        return final_loss
