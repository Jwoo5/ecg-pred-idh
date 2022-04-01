import argparse
import logging
import os
import sys

# should setup root logger before importing any relevant libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import DataLoader

import utils
from .data import FileECGDataset
from models import ConvTransformerModel

from sklearn.metrics import roc_auc_score, roc_curve

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, help="path to data directory", required=True
    )
    parser.add_argument(
        "--label", type=str, default="idh_a", choices=["idh_a", "idh_b", "idh_ab"],
        help="load labels according to this key"
    )
    parser.add_argument(
        "--valid_subset", type=str, default="valid,test",
        help= "comma separated list of data subsets to use for validation (e.g. train, valid, test)"
    )
    parser.add_argument(
        "--bsz", type=int, default=64
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="patient for early stopping, if set <=0 don't stop early"
    )
    parser.add_argument(
        "--save_dir", type=str, default='.'
    )

    # loading pre-trained model
    parser.add_argument(
        "--load_pretrained_weights", type=bool, default=False, action="store_true",
        help="whether to load model weights from pre-trained checkpoint"
    )
    parser.add_argument(
        "--pretrained_model_name", type=str, default="wav2vec2", choices=["wav2vec2"],
        help="pre-trained model name to be loaded. the only available is wav2vec2 currently"
    )
    parser.add_argument(
        "--pretrained_model_path", type=str, default="checkpoint.pt",
        help="path to pre-trained checkpoint"
    )

    # convnets
    parser.add_argument(
        "--extractor_mode", type=str, default="default",
        help="mode for conv feature extractor. default has a single group norm with d"
        "groups in the first conv block, whereas layer_norm layer has layer norms in "
        "every block (menat to use with normalize=True"
    )
    parser.add_argument(
        "--conv_feature_layers", type=str, default="[(256, 2, 2)] * 4",
        help="string describing convolutional feature extraction layers "
        "in form of a python list that contains "
        "[(dim, kernel_size, stride), ...]"
    )
    parser.add_argument(
        "--in_d", type=int, default=12,
        help="input dimension"
    )
    parser.add_argument(
        "--conv_bias", type=bool, default=False,
        help="include bas in conv encoder"
    )
    parser.add_argument(
        "--feature_grad_mult", type=float, default=1.0,
        help="multiply feature extractor var grads by this"
    )

    # transformers
    parser.add_argument(
        "--encoder_layers", type=int, default=12,
        help="num encoder layers in the transformer"
    )
    parser.add_argument(
        "--encoder_embed_dim", type=int, default=768,
        help="encoder embedding dimension"
    )
    parser.add_argument(
        "--encoder_ffn_embed_dim", type=int, default=3072,
        help="encoder embedding dimension for FFN"
    )
    parser.add_argument(
        "--encoder_attention_heads", type=int, default=12,
        help="num encoder attention heads"
    )
    parser.add_argument(
        "--final_dim", type=int, default=0,
        help="project final representations and targets to this many dimensions."
        "set to encoder_embed_dim is <= 0"
    )
    parser.add_argument(
        "--layer_norm_first", type=bool, default=False, action="store_true",
        help="apply layernorm first in the transformer"
    )

    # dropouts
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="dropout probability for the transformer"
    )
    parser.add_argument(
        "--attention_dropout", type=float, default=0.1,
        help="dropout probability for attention weights"
    )
    parser.add_argument(
        "--activation_dropout", type=float, default=0.0,
        help="probability of dropping a transformer layer"
    )
    parser.add_argument(
        "--dropout_input", type=float, default=0.0,
        help="dropout to apply to the input (after feat extr)"
    )
    parser.add_argument(
        "--dropout_features", type=float, default=0.0,
        help="dropout to apply to the features (after feat extr)"
    )

    # masking
    parser.add_argument(
        "--mask_length", type=int, default=10,
        help="mask length"
    )
    parser.add_argument(
        "--mask_prob", type=float, defualt=0.65,
        help="probability of replacing a token with mask"
    )
    parser.add_argument(
        "--mask_selection", type=str, default="static",
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose mask length"
    )
    parser.add_argument(
        "--mask_other", type=float, default=0,
        help="secondary mask argument (used for more complex distributions), "
        "see help in compute_mask_indices"
    )
    parser.add_argument(
        "--no_mask_overlap", type=bool, default=False, action="store_true",
        help="whether to allow masks to overlap"
    )
    parser.add_argument(
        "--mask_min_space", type=int, default=1,
        help="min space between spans (if no overlap is enabled)"
    )
    
    # channel masking
    parser.add_argument(
        "--mask_channel_length", type=int, default=10,
        help="length of the mask for features (channels)"
    )
    parser.add_argument(
        "--mask_channel_prob", type=float, default=0.0,
        help="probability of replacing a feature with 0"
    )
    parser.add_argument(
        "--mask_channel_selection", type=str, default="static",
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose mask length for channel masking"
    )
    parser.add_argument(
        "--mask_channel_other", type=float, default=0,
        help="secondary mask argument (used for more complex distributions), "
        "(deprecated)"
    )
    parser.add_argument(
        "no_mask_channel_overlap", type=bool, default=False, action="store_true",
        help="whether to allow channel masks to overlap"
    )
    parser.add_argument(
        "--mask_channel_min_space", type=int, default=1,
        help="min space between spans (if no overlap is enabled)"
    )
    
    # positional embeddings
    parser.add_argument(
        "--conv_pos", type=int, default=128,
        help="number of filters for convolutional positional embeddings"
    )
    parser.add_argument(
        "--conv_pos_gropus", type=int, default=16,
        help="number of groups for convolutional positional embeddings"
    )

    return parser

def main(args):
    set_struct(vars(args))

    valid_subsets = args.valid_subset.replace(' ', '').split(',')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    dataset = dict()
    ds = FileECGDataset(
        manifest_path=os.path.join(args.data, 'train.tsv'),
        max_sample_size=None,
        min_sample_size=0,
        pad=True,
    )

def set_struct(cfg: dict):
    root = os.path.abspath(
        os.path.dirname(__file__)
    )
    from datetime import datetime
    now = datetime.now()
    from pytz import timezone
    # apply timezone manually
    now = now.astimezone(timezone('Asia/Seoul'))

    output_dir = os.path.join(
        root,
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)

    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train.log'
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = ".config"
    os.mkdir(cfg_dir)
    os.mkdir(cfg['save_dir'])

    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in cfg.items():
            print("{}: {}".format(k, v), file=f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)