import argparse
import logging
import logging.config
import os
import sys

from sklearn.metrics import roc_auc_score, average_precision_score

import pprint

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
from data import FileECGDataset
from models import ConvTransformerModel

from sklearn.metrics import roc_auc_score, roc_curve

def get_parser():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument(
        "--data", type=str, help="path to data manifest directory", required=True
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
        "--lr", type=float, default=0.0005
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="patient for early stopping, if set <=0 don't stop early"
    )
    parser.add_argument(
        "--save_dir", type=str, default='checkpoints'
    )

    # optimizer
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='weight decay in optimizer'
    )

    # criterion
    parser.add_argument(
        '--pos_weight', type=str, default=None,
        help='a weight of positive examples. Must be a vector with length equal to the'
        'number of classes. (e.g., "[3.0, 2.0, ...]")'
    )

    # logging
    parser.add_argument(
        "--log_interval", type=int, default=50,
        help="print log every `--log_interval` step"
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
        "--encoder_layers", type=int, default=2,
        help="num encoder layers in the transformer"
    )
    parser.add_argument(
        "--encoder_embed_dim", type=int, default=256,
        help="encoder embedding dimension"
    )
    parser.add_argument(
        "--encoder_ffn_embed_dim", type=int, default=1024,
        help="encoder embedding dimension for FFN"
    )
    parser.add_argument(
        "--encoder_attention_heads", type=int, default=8,
        help="num encoder attention heads"
    )
    parser.add_argument(
        "--final_dim", type=int, default=0,
        help="project final representations and targets to this many dimensions."
        "set to encoder_embed_dim is <= 0"
    )
    parser.add_argument(
        "--layer_norm_first", default=False, action="store_true",
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
        help="dropout probability after activation in FFN"
    )
    parser.add_argument(
        "--encoder_layerdrop", type=float, default=0.0,
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
        "--mask_prob", type=float, default=0.65,
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
        "--no_mask_overlap", default=False, action="store_true",
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
        "no_mask_channel_overlap", default=False, action="store_true",
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
        "--conv_pos_groups", type=int, default=16,
        help="number of groups for convolutional positional embeddings"
    )

    return parser

def main(args):
    set_struct(vars(args))
    logger.info(pprint.pformat(vars(args)))

    valid_subsets = args.valid_subset.replace(' ', '').split(',')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    dataset = dict()
    ds = FileECGDataset(
        manifest_path=os.path.join(args.data, 'train.tsv'),
        max_sample_size=None,
        min_sample_size=0,
        pad=True,
        label_key=args.label,
    )
    dataset['train'] = DataLoader(
        ds,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=ds.collator
    )
    for valid in valid_subsets:
        ds = FileECGDataset(
            manifest_path=os.path.join(args.data, valid+'.tsv'),
            max_sample_size=None,
            min_sample_size=0,
            pad=True,
            label_key=args.label,
        )
        dataset[valid] = DataLoader(
            ds,
            batch_size=args.bsz,
            shuffle=False,
            collate_fn=ds.collator
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    model = ConvTransformerModel(
        conv_feature_layers=args.conv_feature_layers,
        in_d=args.in_d,
        conv_bias=args.conv_bias,
        feature_grad_mult=args.feature_grad_mult,
        encoder_layers=args.encoder_layers,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_ffn_embed_dim=args.encoder_ffn_embed_dim,
        encoder_attention_heads=args.encoder_attention_heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        encoder_layerdrop=args.encoder_layerdrop,
        dropout_input=args.dropout_input,
        dropout_features=args.dropout_features,
        conv_pos=args.conv_pos,
        conv_pos_groups=args.conv_pos_groups,
        layer_norm_first=args.layer_norm_first,
        num_labels=2 if args.label == 'idh_ab' else 1,
    )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = args.pos_weight
    if pos_weight:
        pos_weight = eval(pos_weight)
        pos_weight = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    logger.info(model)
    logger.info(f"task: {args.label}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    for epoch in range(args.max_epoch):
        logger.info(f"begin training epoch {epoch}")
        logger.info("Start iterating over samples")

        probs = {'train': []}
        truth = {'train': []}
        total_loss = {'train': 0}
        auroc = {'train': 0}
        auprc = {'train': 0}
        for v in valid_subsets:
            probs[v] = []
            truth[v] = []
            total_loss[v] = 0
            auroc[v] = 0
            auprc[v] = 0

        for i, sample in enumerate(dataset['train']):
            model.train()
            criterion.train()
            optimizer.zero_grad()

            sample = utils.prepare_sample(sample)
            net_output = model(**sample["net_input"])
            logits = model.get_logits(net_output)
            targets = model.get_targets(sample, net_output)

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss['train'] += loss.detach()

            if  (i+1) % args.log_interval == 0:
                with utils.rename_logger(logger, 'train'):
                    logger.info(
                        'epoch: {}, update: {:.2f}, avg_loss: {:.3f}'.format(
                            epoch+1, epoch+(i/len(dataset['train'])), total_loss['train'] / (i+1)
                        )
                    )

            with torch.no_grad():
                truth['train'].append(
                    targets.cpu().numpy()
                )
                probs['train'].append(
                    torch.sigmoid(logits).cpu().numpy()
                )
        
        for v in valid_subsets:
            logger.info(f'begin validation on "{v}" subset')

            for sample in dataset[v]:
                model.eval()
                criterion.eval()
                sample = utils.prepare_sample(sample)

                with torch.no_grad():
                    net_output = model(**sample['net_input'])
                    logits = model.get_logits(net_output)
                    if args.label != 'idh_ab':
                        logits = torch.sigmoid(logits).squeeze(-1)
                    targets = model.get_targets(sample, net_output)

                    loss = criterion(logits, targets)
                    total_loss[v] += loss

                    truth[v].append(
                        targets.cpu().numpy()
                    )
                    probs[v].append(
                        torch.sigmoid(logits).cpu().numpy()
                    )
            
            truth[v] = np.concatenate(truth[v], axis=0)
            probs[v] = np.concatenate(probs[v], axis=0)
            auroc[v] = roc_auc_score(y_true=truth[v], y_score=probs[v], average='micro')
            auprc[v] = average_precision_score(y_true=truth[v], y_score=probs[v], average='micro')
            with utils.rename_logger(logger, v):
                logger.info(
                    'epoch: {}, {}_loss: {:.3f}, {}_auroc: {:.3f}, {}_auprc: {:.3f}'.format(
                        epoch+1, v, total_loss[v] / len(dataset[v]), v, auroc[v], v, auprc[v]
                    )
                )

        truth['train'] = np.concatenate(truth['train'], axis=0)
        probs['train'] = np.concatenate(probs['train'], axis=0)
        auroc['train'] = roc_auc_score(y_true=truth['train'], y_score=probs['train'], average='micro')
        auprc['train'] = average_precision_score(y_true=truth['train'], y_score=probs['train'], average='micro')
        with utils.rename_logger(logger, 'train'):
            logger.info(f'end of epoch {epoch+1} (average epoch stats below)')
            logger.info(
                'epoch: {}, train_loss: {:.3f}, train_auroc: {:.3f}, train_auprc: {:.3f}'.format(
                    epoch+1, total_loss['train'] / len(dataset['train']), auroc['train'], auprc['train']
                )
            )
        
        should_stop = utils.should_stop_early(args.patience, auroc['valid'])
        if utils.should_stop_early.best == auroc:
            state_dict = {
                'model': model.state_dict(),
                'epoch': epoch+1,
                'valid_auroc': auroc['valid'],
                'loss': total_loss['valid'] / len(dataset['valid'])
            }
            torch.save(
                state_dict,
                os.path.join(args.save_dir, 'checkpoint_best.pt')
            )
        
        if should_stop:
            logger.info(
                f"early stop since valid performance hasn't improved for last {args.patience} runs"
            )
            break

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