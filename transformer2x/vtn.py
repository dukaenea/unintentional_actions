# code from https://github.com/bomri/SlowFast/blob/master/slowfast
# @Author: Enea Duka
# @Date: 5/3/21
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_setup import logger
# from models.prototypical_memory import PrototypicalMemory
# from utils.util_functions import RegressionLoss
# from models.vit import create_vit_model
from torchcrf import CRF
from utils.arg_parse import opt
from itertools import chain
from models.res3d_18 import R3D18

class VTNLongformerModel(transformers.LongformerModel):
    def __init__(self,
                 embed_dim=768,
                 max_positions_embedding=2 * 60 * 60,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 attention_mode='sliding_chunks',
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):
        self.config = transformers.LongformerConfig()
        self.config.attention_mode = attention_mode
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_dilation = [1, ] * num_attention_heads
        self.config.attention_window = [256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_positions_embedding
        self.config.hidden_size = embed_dim
        super(VTNLongformerModel, self).__init__(self.config, add_pooling_layer=False)
        self.embeddings.word_embeddings = None

def pad_to_window_size_local(input_ids, attention_mask, position_ids, one_sided_window_size, pad_token_id, pure_nr_frames):
    w = 2 * one_sided_window_size
    seq_len = input_ids.size(1)
    padding_len = (w - seq_len % w) % w
    input_ids = F.pad(input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id).permute(0, 2, 1)
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
    def __init__(self, local=True, use_bn=False):
        super(VTN, self).__init__()
        self._build_model(local, use_bn)

    def _build_model(self, local, use_bn):
        self.cls_token = nn.Parameter(torch.randn(1, 1, opt.embed_dim))
        self.lin_projector = nn.Linear(3*opt.cuboid_resolution**3, opt.embed_dim)
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
            hidden_dropout_prob=opt.hidden_dropout_prob
        )
        if use_bn:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(opt.hidden_dim),
                nn.Linear(opt.hidden_dim, opt.mlp_dim),
                # nn.BatchNorm1d(opt.mlp_dim),
                nn.GELU(),
                nn.Dropout(opt.mlp_dropout),
                nn.Linear(opt.mlp_dim, opt.num_classes)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(opt.hidden_dim),
                nn.Linear(opt.hidden_dim, opt.mlp_dim),
                nn.GELU(),
                nn.Dropout(opt.mlp_dropout),
                nn.Linear(opt.mlp_dim, opt.num_classes)
            )
        self.local = local

    def forward(self, x, position_ids, speed_labels=None, pure_nr_frames=None, num_clips=None,
                multi_scale=True, video_level_pred=False, return_features=False):
        try:
            B, D, E = x.shape
        except Exception as e:
            print(x.shape)
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
            pure_nr_frames
        )
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(input_ids = None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=True,
                                  return_dict=True)

        # if the model is local, return the global representation for the clip
        x = x['last_hidden_state']
        if self.local and multi_scale:
            x = x[:, 0]
            return x
        elif not self.local and multi_scale and not video_level_pred:
            # if global, we need the representation for all the clips in the video
            xs = []
            for idx, pnf in enumerate(pure_nr_frames):
                xs.append(x[idx][:pnf])
            x = torch.cat(xs, dim=0)
            if return_features:
                return x
        elif (self.local and not multi_scale) or (not self.local and video_level_pred):
            x = x[:, 0]

        x_mlp = self.mlp_head(x)
        return x_mlp

    def reinit_mlp(self, num_classes):
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            # nn.BatchNorm1d(opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(opt.mlp_dim, num_classes)
            )
        self.mlp_head.cuda()


    def freeze(self):
        for name, child in self.named_children():
            if name == 'temporal_encoder':
                for param in child.parameters():
                    param.requires_grad = False


class GlobalVTN(nn.Module):
    def __init__(self, pretrained_frames):
        super(GlobalVTN, self).__init__()
        opt.num_classes = 7
        if opt.backbone == 'vit_longformer':
            self.local_model = VTN(local=True, use_bn=True)
        elif opt.backbone == 'r3d_18':
            self.local_model = R3D18(pretrain_backbone=False, local=True)
        if pretrained_frames:
            self.local_model.cuda()
            self.local_model = nn.DataParallel(self.local_model)
            saved_model = torch.load(opt.vtn_ptr_path if opt.backbone == 'vit_longformer' else opt.resnet_ptr_path)
            model_dict = saved_model['state_dict']
            self.local_model.load_state_dict(model_dict, strict=False)
            self.local_model = self.local_model.module
        # if opt.backbone == 'r3d_18':
        #     self.local_model.remove_mlp()

        opt.num_classes = 7
        self.global_model = VTN(local=False, use_bn=False)
        if opt.use_crf:
            self.crf = CRF(num_tags=3, batch_first=True)

    def _local_forward(self, x, position_ids, pure_nr_frames, multi_scale):
        clip_encodings = self.local_model(x.squeeze(), position_ids, None, pure_nr_frames, multi_scale=multi_scale)
        return clip_encodings, pure_nr_frames

    def _local_forward_r3d(self, x):
        clip_encodings = self.local_model(x.permute(0, 2, 1, 3, 4))
        return clip_encodings

    def forward(self, x, position_ids, speed_labels=None,
                pure_nr_frames=None, labels=None, num_clips=None,
                local=False, for_crf=False, multi_scale=True,
                video_level_pred=False, return_features=False):

        if for_crf:
            # create the mask for the CRF layer
            crf_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.uint8).to(x.device)
            mask = (x == 0)[:, :, 0]
            crf_mask[mask] = 0
            if self.training:
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(0)
                # if opt.crf_margin_probs:
                #     x = x.permute(1, 0, 2)
                #     crf_mask = crf_mask.permute(1, 0)
                #     probs = self.crf.compute_marginal_probabilities(x, crf_mask)
                #     return probs
                # else:
                loss = -self.crf.forward(x, labels, mask=crf_mask)
                return loss
            else:
                if opt.crf_margin_probs:
                    x = x.permute(1, 0, 2)
                    crf_mask = crf_mask.permute(1, 0)
                    probs = self.crf.compute_marginal_probabilities(x, crf_mask)
                    return probs
                else:
                    loss = -self.crf.forward(x, labels, mask=crf_mask)
                    mls = self.crf.decode(x, mask=crf_mask)
                    return torch.tensor(list(chain.from_iterable(mls))).to(x.device), loss
        else:
            if local:
                if opt.backbone == 'vit_longformer':
                    clip_encodings, pure_nr_frames = self._local_forward(x, position_ids, pure_nr_frames, multi_scale)
                    return clip_encodings, pure_nr_frames
                elif opt.backbone == 'r3d_18':
                    clip_encodings = self._local_forward_r3d(x)
                    return clip_encodings
            else:
                outs = self.global_model(x, position_ids, pure_nr_frames=pure_nr_frames, num_clips=num_clips, video_level_pred=video_level_pred, return_features=return_features)
                return outs


    def reinit_mlp(self, num_classes):
        self.global_model.reinit_mlp(num_classes)
        if opt.backbone == 'vit_longformer':
            self.local_model.reinit_mlp(num_classes)
        elif opt.backbone == 'r3d_18':
            self.local_model.remove_mlp()




def get_froze_trn_optimizer(model):
    if opt.optim == 'adam':
        optimizer = torch.optim.AdamW([{'params': model.module.mlp_head.parameters(), 'lr': opt.lr},
                                     {'params': model.module.temporal_encoder.parameters(), 'lr': opt.lr * opt.backbone_lr_factor}],
                                      weight_decay=opt.weight_decay)
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.module.mlp_head.parameters(), 'lr': opt.lr},
                                     {'params': model.module.temporal_encoder.parameters(), 'lr': 0}],
                                    momentum=opt.momentum, weight_decay=opt.weight_decay)

    return optimizer

def part_freeze_vit(model):
    for param in model.named_parameters():
        if 'crf' in param[0] or 'mlp_head' in param[0]:
            print(param[0])
        else:
            param[1].requires_grad = False
            print(param[0])

def create_model_trn_2x(num_classes=None, pretrained=False, pretrain_scale='frame+clip'):
    model = GlobalVTN(pretrained and pretrain_scale == 'frame')
    if num_classes is not None and not pretrained:
        model.reinit_mlp(num_classes)
    model.cuda()
    if opt.gpu_parallel:
        model = nn.DataParallel(model)
        if pretrained and pretrain_scale == 'frame+clip':
            saved_model = torch.load(opt.vtn_ptr_path)
            model_dict = saved_model['state_dict']
            model.load_state_dict(model_dict, strict=False)
            if opt.use_memory:
                model.module.memory.memory = saved_model['memory_state']
        if pretrained:
            # model.module.freeze()
            if num_classes is not None:
                model.module.reinit_mlp(num_classes)
            model.module.cuda()
            if opt.gpu_parallel:
                model = nn.DataParallel(model.module)
                # else:
                #     model.freeze()
                #     model.reinit_mlp(num_classes)
    # part_freeze_vit(model)

    if opt.optim == 'adam':
        if opt.rep_learning:
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=opt.weight_decay)
        else:
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.optim == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    if opt.rep_learning:
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([19670/18069, 19670/4137, 19670/19670]).cuda())
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([83023/61320, 83023/16097, 83023/83023]).cuda())
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([27311/10707, 27311/3846, 27311/27311]).cuda()) # this is for non-overlapping 1.5 secs in the future.
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([22918/14755, 22918/4191, 22918/22918]).cuda()) # this is for non-overlapping 0.5 secs in the future.
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([25106/12697, 25106/4061, 25106/25106]).cuda()) # this is for non-overlapping 1.0 secs in the future.
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([29370/8959, 29370/3535, 29370/29370]).cuda()) # this is for non-overlapping 2.0 secs in the future.


    logger.debug(str(model))
    logger.debug(str(optimizer))
    logger.debug(str(loss))
    return model, optimizer, loss