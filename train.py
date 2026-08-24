"""
Training entry point for GADAN with a configurable text encoder and GSBI.

Usage:
    # BiLSTM + GSBI (default)
    python train.py --dataset_file rsvg --output_dir outputs/gadan ...

    # RoBERTa + GSBI
    python train.py --text_encoder_type roberta --lr_text_encoder 1e-5 ...

    # Frozen RoBERTa + GSBI
    python train.py --text_encoder_type roberta_frozen --freeze_text_encoder ...

    # BERT-tiny + GSBI
    python train.py --text_encoder_type bert_tiny --text_encoder_path google/bert_uncased_L-2_H-128_A-2 ...

Differences from the baseline:
    - Imports GADAN from models.gadan
    - Exposes --text_encoder_type to select the text encoder
    - Exposes --gsbi_d_geo to control the geometry embedding dimension
    - All other hyper-parameters are shared with the standard baseline
"""

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import torch_patch
import datasets.samplers as samplers
import opts
import util.misc as utils
from datasets import build_dataset
from datasets.coco_eval import CocoEvaluator
from engine import train_one_epoch
from models.gadan import build as build_model
from tools.load_pretrained_weights import pre_trained_model_to_finetune
from torch.utils.data import DataLoader
from util.checkpoint import load_model_state_strict


PAPER_TRAINING_CONFIG = {
    "lr": 1e-4,
    "lr_backbone": 5e-5,
    "lr_text_encoder": 1e-5,
    "weight_decay": 5e-4,
    "epochs": 70,
    "lr_drop": [60],
    "max_size": 640,
    "backbone": "resnet50",
    "hidden_dim": 256,
    "num_feature_levels": 4,
    "nheads": 8,
    "dec_layers": 4,
    "num_queries": 10,
    "bilstm_embed_dim": 300,
    "bilstm_hidden_dim": 128,
    "bilstm_num_layers": 2,
    "bilstm_dropout": 0.1,
    "gsbi_d_geo": 64,
}


def validate_paper_training_config(args):
    mismatches = []
    for name, expected in PAPER_TRAINING_CONFIG.items():
        actual = getattr(args, name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError(
            "Training configuration differs from the paper:\n  "
            + "\n  ".join(mismatches)
        )


def main(args):
    os.environ["MDETR_CPU_REDUCE"] = "1"

    args.masks = False
    assert args.dataset_file == "rsvg"
    validate_paper_training_config(args)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(name, name_keywords):
        return any(keyword in name for keyword in name_keywords)

    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_text_encoder_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad
            ],
            "lr": args.lr_text_encoder,
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.dataset_file != "all":
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
        print('trainset:', len(dataset_train))
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]
        dataset_train = torch.utils.data.ConcatDataset(
            [build_dataset(name, image_set="train", args=args) for name in dataset_names]
        )

    print('trainset:', len(dataset_train))
    print("\nTrain dataset sample number: ", len(dataset_train))
    print("\n")

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    def build_evaluator_list(base_ds, dataset_name):
        evaluator_list = []
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")
        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        return evaluator_list

    output_dir = Path(args.output_dir)
    if args.resume:
        print("Resume from {}".format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if "args" not in checkpoint:
            raise RuntimeError(
                "Resume checkpoint has no saved training configuration."
            )
        validate_paper_training_config(checkpoint["args"])
        load_model_state_strict(model_without_ddp, checkpoint)
        if (
            not args.eval
            and 'optimizer' in checkpoint
            and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as file_handle:
                file_handle.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_bilstm_gsbi_parser():
    parser = argparse.ArgumentParser(
        'GADAN training and evaluation script',
        parents=[opts.get_args_parser()],
    )
    # Text encoder selection
    parser.add_argument('--text_encoder_type', default='bilstm', type=str,
                        choices=['bilstm', 'roberta', 'roberta_frozen', 'bert_tiny'],
                        help='Text encoder type: bilstm, roberta, roberta_frozen, bert_tiny')
    # BiLSTM text encoder args
    parser.add_argument('--bilstm_embed_dim', default=300, type=int,
                        help='BiLSTM token embedding dimension')
    parser.add_argument('--bilstm_hidden_dim', default=128, type=int,
                        help='BiLSTM hidden size per direction')
    parser.add_argument('--bilstm_num_layers', default=2, type=int,
                        help='BiLSTM layer count')
    parser.add_argument('--bilstm_dropout', default=0.1, type=float,
                        help='BiLSTM dropout rate')
    # GSBI args
    parser.add_argument('--gsbi_d_geo', default=64, type=int,
                        help='Geometry embedding dimension for GSBI module')
    return parser


if __name__ == '__main__':
    parser = get_bilstm_gsbi_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
