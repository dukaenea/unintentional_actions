# code from https://github.com/bomri/SlowFast/blob/master/slowfast
# @Author: Enea Duka
# @Date: 5/3/21
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_setup import logger
from models.prototypical_memory import PrototypicalMemory
from utils.util_functions import RegressionLoss
from models.vit import create_vit_model

from utils.arg_parse import opt

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
        super(VTNLongformerModel, self).__init__(self.config, add_pooling_layer=True)
        self.embeddings.word_embeddings = None

def pad_to_window_size_local(input_ids, attention_mask, position_ids, one_sided_window_size, pad_token_id, pure_nr_frames):
    w = 2 * one_sided_window_size
    seq_len = input_ids.size(1)
    padding_len = (w - seq_len % w) % w
    input_ids = F.pad(input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id).permute(0, 2, 1)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)
    position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)

    # if pure_nr_frames is not None:
    #     for idx, nr_frames in enumerate(pure_nr_frames):
    #         try:
    #             attention_mask[idx, nr_frames:] = 0
    #             position_ids[idx, nr_frames:] = 1
    #         except Exception as e:
    #             pass
    return input_ids, attention_mask, position_ids


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self._build_model()

    def _build_model(self):
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

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.mlp_dim),
            # nn.BatchNorm1d(opt.mlp_dim),
            nn.GELU(),
            nn.Dropout(opt.mlp_dropout),
            nn.Linear(opt.mlp_dim, opt.num_classes),
        )


    def forward(self, x, position_ids, speed_labels=None, pure_nr_frames=None, return_features=False):
        # if opt.use_bbone:
        #     x_feats = []
        #     for clip in x:
        #         clip = clip.permute(1, 0, 2, 3)
        #         clip_feats = self.bbone(clip)
        #         x_feats.append(clip_feats)
        #     x = torch.stack(x_feats)
        if opt.spat_temp:
            # unfold the video into cuboids
            # find the num of cuboids along the temporal axis
            n, c, t, w, h = x.shape
            temp_cuboids = t // opt.cuboid_resolution
            spat_cuboids = w // opt.cuboid_resolution
            x = x[:, :, 0:(temp_cuboids * opt.cuboid_resolution) + 1, :, :]
            # x = x.unfold(2, temp_cuboids, temp_cuboids)
            x = (
                x.unfold(2, temp_cuboids, temp_cuboids)
                .unfold(3, spat_cuboids, spat_cuboids)
                .unfold(4, spat_cuboids, spat_cuboids)
                .contiguous()
            )
            x = x.reshape(n, c, -1, temp_cuboids * spat_cuboids ** 2)
            x = x.permute(0, 3, 2, 1).contiguous()
            # not 100% sure about this, check later if it's not working
            x = x.reshape(x.size(0), temp_cuboids * spat_cuboids ** 2, -1)

            position_ids = torch.tensor(list(range(0, x.shape[1])), device=x.device) \
                .expand(1, x.shape[1]) \
                .repeat(x.shape[0], 1)
            x = self.lin_projector(x)
        else:
            # at this point the samples have passed from the spatial component
            # and from the transformation pipeline

            # the input should be of the shape [B, T, C, H, W]
            B, C, T = x.shape
            # x = x.permute(0, 2, 1)
            # flatten each frame and treat is as the frame embedding to feed to the model
            # x = x.reshape(B, T, -1)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_token = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        cls_attn = torch.ones((1)).expand(B, -1).to(x.device)
        # attention_mask = torch.cat((attention_mask, cls_attn), dim=1)
        # attention_mask[:, 0] = 2
        x, attention_mask, position_ids = pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id,
            pure_nr_frames
        )
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        # token_type_ids[:, 0] = 1

        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        # position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(input_ids = None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=True,
                                  return_dict=True)
        if return_features:
            return x['pooler_output']
            x = x['last_hidden_state']
            x = x[:, :pure_nr_frames]
            return x
        # x = x[:, 0]
        x = x['pooler_output']
        # x = x.mean(1)
        if opt.mmargin_loss:
            x_mlp = self.mlp_head(x)
            return x, x_mlp
        if opt.use_memory:
            x = self.memory(x)
        if opt.create_memory:
            return x
        # if opt.rep_learning and not self.training:
        #     x_mlp = self.mlp_head_secondary(x)
        else:
            x_mlp = self.mlp_head(x)
        if opt.consist_lrn:
            return x_mlp, x
        return x_mlp

    def reinit_mlp(self, num_classes):
        if opt.task == 'classification':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(opt.hidden_dim),
                nn.Linear(opt.hidden_dim, opt.mlp_dim),
                nn.BatchNorm1d(opt.mlp_dim),
                nn.GELU(),
                nn.Dropout(opt.mlp_dropout),
                # nn.LayerNorm(opt.mlp_dim * 2),
                # nn.Linear(opt.mlp_dim * 2, opt.mlp_dim),
                # nn.GELU(),
                # nn.Dropout(opt.mlp_dropout),
                nn.Linear(opt.mlp_dim, opt.num_classes),
                # nn.Sigmoid()
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(opt.hidden_dim),
                nn.Linear(opt.hidden_dim, opt.mlp_dim),
                nn.GELU(),
                nn.Dropout(opt.mlp_dropout),
                # nn.LayerNorm(opt.mlp_dim * 2),
                # nn.Linear(opt.mlp_dim * 2, opt.mlp_dim),
                # nn.GELU(),
                # nn.Dropout(opt.mlp_dropout),
                nn.Linear(opt.mlp_dim, 1),
                nn.Sigmoid()
            )
        self.mlp_head.cuda()


    def freeze(self):
        for name, child in self.named_children():
            if name == 'temporal_encoder':
                for param in child.parameters():
                    param.requires_grad = False

    def update_memory_weights(self, weight, dataset, filename):
        self.memory.update_memory_weights(weight, dataset, filename)

    def update_memory(self, features, dataset):
        self.memory.update_memory(features, dataset)

    def sharpen_normalize_memory_weights(self):
        self.memory.sharpen_normalize_weights()

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
                model_dict = saved_model['state_dict']
            except KeyError:
                model_dict = saved_model['vtn_state_dict']
            model.load_state_dict(model_dict, strict=True)
            if opt.use_memory:
                model.module.memory.memory = saved_model['memory_state']
            if opt.gpu_parallel:
                # model.module.freeze()
                if num_classes is not None:
                    model.module.reinit_mlp(num_classes)
                model.module.cuda()
                if opt.gpu_parallel:
                    model = nn.parallel.DistributedDataParallel(model.module)
            else:
                model.freeze()
                model.reinit_mlp(num_classes)

    if opt.optim == 'adam':
        if opt.rep_learning:
            optimizer = torch.optim.AdamW([{'params': model.module.mlp_head.parameters(), 'lr': opt.lr},
                                           {'params': model.module.temporal_encoder.parameters(), 'lr': opt.lr}],
                                          weight_decay=opt.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.optim == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
       # optimizer = torch.optim.SGD([{'params': model.module.mlp_head.parameters(), 'lr': opt.lr},
       #                              {'params': model.module.temporal_encoder.parameters(), 'lr': opt.lr*0.1}],
       #                             momentum=opt.momentum, weight_decay=opt.weight_decay)

    # if opt.dataset == 'kinetics':
    if opt.task == 'classification':
        # loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.25, 0.4, 0.25]).cuda())ss
        loss = nn.CrossEntropyLoss()
        # loss = nn.BCELoss()
    elif opt.task == 'regression':
        # loss = nn.MSELoss(reduction='sum')
        loss = nn.SmoothL1Loss(beta=0.01, reduction='sum')
        # loss = RegressionLoss()

    # elif opt.dataset == 'oops':
    #     loss = nn.BCEWithLogitsLoss()
    logger.debug(str(model))
    logger.debug(str(optimizer))
    logger.debug(str(loss))
    return model, optimizer, loss