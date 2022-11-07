#!/usr/bin/env python

""" Main parameters for the project. Other parameters are dataset specific.
"""

__author__ = "Anna Kukleva"
__date__ = "February 2020"


import argparse
import sys
import yaml

parser = argparse.ArgumentParser()
if len(sys.argv) > 1:
    yml_path = sys.argv[1]

#######################################################################################
### DATA
parser.add_argument(
    "--dataset",
    default="kt",
    help="bf: breakfast dataset"
    "kt: kinetics dataset"
    "hmdb : hmdb dataset"
    "ucf: ucf101 dataset",
)
parser.add_argument("--data_root", default="")
parser.add_argument("--gt_dir", default="")
parser.add_argument("--feat_dir", default="")
parser.add_argument("--one_plus2_model_pref", default="/BS/kukleva/work/models/")
parser.add_argument(
    "--one_plus2_model",
    default="",
    help="pretrained just on sport1m: : r25d34_sports1m.pth"
    "pretrained on sport1m and base kinetics: r25d34_kinetics100_save_1.pth (89.34),"
    "v0.9060_ep1.pth.tar,"
    "v0.9216_ep14.pth.tar  - attention model without vt",
)
parser.add_argument("--aggregate", default="mean")
parser.add_argument("--temp_embed", default=False)


#######################################################################################
### KINETICS
parser.add_argument(
    "--feat_type",
    default="sp+kin",
    help="preextracted features:"
    "sp          -- just on the sport1m dataset"
    "sp+kin      -- on sport1m and kinetics 100 base classes"
    "sp1m_kt_spat9060 -- spatial features on sport1m and kin100base"
    "s3d -- features s3d from antoine"
    "video       -- pure videos",
)
parser.add_argument(
    "--spatial_size",
    default=112,
    type=int,
    help="spatial crop size for input video frames",
)
parser.add_argument(
    "--video_fps", default=25, type=int, help="unify fps for all videos"
)
parser.add_argument(
    "--spat_crop",
    default="central",
    help="if to perform central crop or not for augmentation",
)
parser.add_argument("--spat_scale", default=False, type=bool)
parser.add_argument(
    "--h_flip",
    default=False,
    type=bool,
    help="if to perform horizontal flip of the frames for augmentation",
)
parser.add_argument(
    "--inverse_colors", default=False, help="inverse frames colors for augmentation"
)
parser.add_argument("--extract_features", default=False, type=bool)
parser.add_argument("--n_base_classes", default=64, type=int)

#######################################################################################
### OOPS!!
parser.add_argument("--val", default=False, type=bool)
parser.add_argument(
    "--fails_path",
    default="/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video",
    type=str,
)
parser.add_argument("--fails_only", default=True, type=bool)
parser.add_argument("--workers", default=0, type=int)
parser.add_argument("--all_fail_videos", default=True, type=bool)
parser.add_argument("--cache_dataset", default=True, type=bool)
parser.add_argument("--fails_action_split", default=False, type=bool)
parser.add_argument("--balance_fails_only", default=False, type=bool)
parser.add_argument("--fps_list", default=[16], nargs="+")
parser.add_argument("--step_between_clips_sec", default=0.25, type=float)
parser.add_argument("--anticipate_label", default=0, type=float)
parser.add_argument("--remove_fns", default="None", type=str)
parser.add_argument("--sample_all_clips", default=True, type=bool)
parser.add_argument("--clips_per_video", default=10, type=int)
parser.add_argument(
    "--kinetics_path",
    default="/BS/unintentional_actions/nobackup/kinetics400/metadata",
    type=str,
)
parser.add_argument("--frames_per_clip", default=16, type=int)
parser.add_argument("--clip_interval_factor", default=1, type=int)
parser.add_argument(
    "--dataset_path", default="/BS/unintentional_actions/nobackup/oops", type=str
)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--sample_videos", default=False, type=bool)
parser.add_argument("--selfsup_loss", type=str, default=None)
parser.add_argument("--load_videos", type=bool, default=False)
#######################################################################################
### GENERAL
parser.add_argument("--model_name", default="")
parser.add_argument("--sfx", default="")
parser.add_argument("--pfx", default="")
parser.add_argument("--crt_weight_training", default=False)
parser.add_argument("--gpu_parallel", default=False, type=bool)
parser.add_argument("--backbone_lr_factor", default=None, type=float)
parser.add_argument("--cos_decay_lr_factor", default=None, type=float)
parser.add_argument("--pretrained", default=False, type=bool)
parser.add_argument("--backbone", default="vit_longformer")

#######################################################################################
### TEMPORAL_REGULARITY
parser.add_argument("--temp_learning_dataset_name", default="ped", type=str)
parser.add_argument("--num_in_channels", default=10, type=int)
parser.add_argument("--ptr_tmpreg_model_path", default=None, type=str)

#######################################################################################
### ACTION_CLASSIFICATION
parser.add_argument("--act_backbone", default="None", type=str)
parser.add_argument("--task", default="classification", type=str)
parser.add_argument("--use_memory", default=False, type=str)
parser.add_argument("--mmargin_loss", default=False, type=str)
#######################################################################################
### UNINTENTIONAL_ACTIONS
parser.add_argument("--unint_act_backbone", default=None, type=str)


#######################################################################################
### LONGFORMER
parser.add_argument("--embed_dim", default=2048, type=int)
parser.add_argument("--max_positions_embedding", default=7200, type=int)
parser.add_argument("--num_attention_heads", default=16, type=int)
parser.add_argument("--num_hidden_layers", default=3, type=int)
parser.add_argument("--attention_mode", default="sliding_chunks", type=str)
parser.add_argument("--pad_token_id", default=-1, type=int)
parser.add_argument("--attention_window", default=[32, 32, 32], nargs="+")
parser.add_argument("--intermediate_size", default=4096, type=int)
parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
parser.add_argument("--hidden_dim", default=2048, type=int)
parser.add_argument("--mlp_dropout", default=0.2, type=float)
parser.add_argument("--mlp_dim", default=1024, type=int)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--num_classes_ptr", default=10, type=int)
parser.add_argument("--vtn_ptr_path", default="", type=str)
parser.add_argument("--spat_temp", default=True, type=bool)
parser.add_argument("--cuboid_resolution", default=8, type=int)
parser.add_argument("--use_bbone", default=False, type=bool)
parser.add_argument("--use_crf", default=False, type=bool)
parser.add_argument("--crf_margin_probs", default=False, type=bool)
parser.add_argument("--multi_scale", default=False, type=bool)
parser.add_argument("--resnet_ptr_path", default="", type=str)


#######################################################################################
### REPRESENTATION_LEARNING
parser.add_argument("--rep_backbone", default="r2plus1d", type=str)
parser.add_argument("--rep_data_level", default="features", type=str)
parser.add_argument("--trans_prob", default=0.5, type=int)
parser.add_argument("--lr_scheduler", default="step", type=str)
parser.add_argument("--speed_and_motion", default=False, type=bool)
parser.add_argument("--consist_lrn", default=False, type=bool)
parser.add_argument("--dataset_subset", default="", type=str)
parser.add_argument("--tag", default="", type=str)
parser.add_argument("--contrastive_learning", default=False, type=bool)
parser.add_argument("--create_memory", default=False, type=bool)
parser.add_argument("--rep_learning", default=False, type=bool)
parser.add_argument("--transformation_groups", default="speed&motion", type=bool)
parser.add_argument(
    "--transformations_list",
    default=[
        "normal",
        "speedx2",
        "speedx3",
        "speedx4",
        "shuffle",
        "warp",
        "random_point_speedup",
        "double_flip",
    ],
    nargs="+",
)

#######################################################################################
### FEATURE_AUGMENTATION

parser.add_argument(
    "--spatial_augmentation",
    default=False,
    type=bool,
    help="augment spatially features when training classifier",
)
parser.add_argument(
    "--temporal_augmentation",
    default=False,
    type=bool,
    help="augment temporally features when training classifier",
)
parser.add_argument(
    "--two_stream",
    default=False,
    type=bool,
    help="augment features when training classifier",
)
parser.add_argument(
    "--add_noise",
    default=False,
    type=bool,
    help="augment features by adding gaussian noise when " "training classifier",
)
parser.add_argument(
    "--nn_lerp",
    default=False,
    type=bool,
    help="augment features by interpolating between nearest"
    "neighbours when training classifier",
)
parser.add_argument(
    "--linear_delta",
    default=False,
    type=bool,
    help="augment features by calculating difference of nearest"
    "neighbours when training classifier",
)
parser.add_argument(
    "--spat_augms", default="", type=str, help="list of spatial augmentations"
)
parser.add_argument(
    "--temp_augms", default="", type=str, help="list of temporal augmentations"
)

parser.add_argument(
    "--tsn_encoder",
    default="",
    type=str,
    help="use tsn encoder before feeding to classifier",
)
parser.add_argument(
    "--use_dtn",
    default=False,
    type=bool,
    help="use Detail Transfer Network to synthesize new metadata",
)

#######################################################################################
### GENERAL HYPERPARAMS
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=3e-3, type=float)
parser.add_argument("--cl_lr", default=3e-3, type=float)
parser.add_argument("--lr_stem", default=3e-5, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument(
    "--weight_decay", default=1e-4, type=float, help="usually it is 1e-4"
)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--num_workers", default=10, type=int)
parser.add_argument("--device", default="cuda", help="cuda | cpu")
parser.add_argument(
    "--optim", default="adam", help="optimizer for the network: adam | sgd"
)
parser.add_argument("--momentum", default=0)
parser.add_argument("--cl_momentum", default=0)
parser.add_argument("--ep_optim_base", default=0)
parser.add_argument("--use_tqdm", default=True)
parser.add_argument("--debug", default=False)


#######################################################################################
### VALIDATION & TEST
parser.add_argument("--test_freq", default=1, type=int)
parser.add_argument("--test_val", default=True, type=bool)
parser.add_argument(
    "--test", default=False, type=bool, help="to test on the test splits"
)

#######################################################################################
### STORAGE
parser.add_argument("--storage", default="/BS/unintentional_actions/work/storage")
parser.add_argument(
    "--save_model",
    default=0,
    type=int,
    help="how often save attention, 0 - do not save at all",
)
parser.add_argument("--pretrain", default=False)

#######################################################################################
### LOGS
parser.add_argument(
    "--log_mode", default="DEBUG", help="DEBUG | INFO | WARNING | ERROR | CRITICAL"
)
parser.add_argument("--log_save_dir", default="/BS/unintentional_actions/nobackup/logs")
parser.add_argument("--log_name", default="")
parser.add_argument("--debug_freq", default=1, type=int)


#######################################################################################
### VISDOM
parser.add_argument("--viz", default=True, type=bool)
parser.add_argument("--viz_env", default="main")
parser.add_argument("--env_pref", default="")
parser.add_argument("--sanity_chk", default=False, type=bool)
parser.add_argument("--viz_freq", default=1, type=int)
parser.add_argument(
    "--viz_meta",
    default=True,
    type=bool,
    help="unable plotting within inner loop during meta-learning",
)


opt, unknown = parser.parse_known_args()


if len(sys.argv) > 1:
    with open(yml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        config = dict()
        for key in data.keys():
            config = dict(config, **data[key])

        config_keys = list(config.keys())

        for attr, value in opt.__dict__.items():
            if attr in config_keys:
                setattr(opt, attr, config[attr])
