import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
import math
import wandb
import scanpy as sc
import pandas as pd
from pathlib import Path
from typing import Literal, Union
import numpy as np
from anndata import AnnData
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
# import pytorch_lightning as pl
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.data_loader_pretrain import paProteinDataset, paTESTProteinDataset_citeseq, paTESTProteinDataset_cytof
from utils import (
    select_device,
    apply_noise,
    increment_path,
    torch_distributed_zero_first,
    reduce_value,
    save_checkpoint,
    epoch_time,
)
from utils import (
    count_parameters,
    apply_noise,
    save_checkpoint,
    get_std_logging,
    mask_tensor,
    setup_seed,
    mask_generator,
    pretext_generator,
)
import argparse
from tqdm import tqdm
from model.model_citeseq_notech import scTrans
from model.configuration_hyena import HyenaConfig

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
# LOCAL_RANK = 0
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))
print("LOCAL RANK:", LOCAL_RANK)
print("RANK:", RANK)
print("WORLD_SIZE:", WORLD_SIZE)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/RNA/cell_gene",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./result2024",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument("--freeze_layers", action="store_true")
    parser.add_argument("--syncBN", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start_epochs", default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--device", type=str, default=8, help="device = 'cpu' or '0' or '0,1,2,3'"
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument("--hyena_dropout", type=float, default=0.0)
    parser.add_argument(
        "--enc_vocab_len", type=int, default=6427, help="vocab length of encoder"
    )
    parser.add_argument(
        "--dec_vocab_len", type=int, default=6427, help="vocab length of decoder"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1000, help="sequence length"
    )
    parser.add_argument(
        "--enc_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--dec_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=32, help="latend dimension of each token"
    )
    parser.add_argument(
        "--dec_dim", type=int, default=32, help="latend dimension of each token"
    )
    parser.add_argument(
        "--position_emb_dim",
        type=int,
        default=5,
        help="dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands",
    )
    parser.add_argument("--wandb_name", type=str, default="test", help="wanbd name")

    args = parser.parse_args()
    config = vars(args)
    return config


def main(config):
    print(config)
    pl.seed_everything(config["seed"])

    save_dir = Path(str(increment_path(Path(config["save_path"]), mkdir=True)))
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Create data
    all_data = paProteinDataset(
        data_dir="./10M_pro_databanks/",
    )
    n_train = int(len(all_data) * 0.9)
    train_data, valid_data = torch.utils.data.random_split(
        all_data, [n_train, len(all_data) - n_train]
    )
    test_data_cite = paTESTProteinDataset_citeseq(
        data_dir="./10M_citeseq_databanks",
    )
    test_data_cytof = paTESTProteinDataset_citeseq(
        data_dir="./10M_cytof_databanks",
    )

    train_sampler = None
    valid_sampler = None
    test_sampler = None

    nw = 8
    print("----------nw", nw)
    train_dataloader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=train_sampler is None,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=valid_sampler,
        # collate_fn=collator,
        drop_last=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_dataloader_citeseq = DataLoader(
        test_data_cite,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        # collate_fn=collator,
        drop_last=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_dataloader_cytof = DataLoader(
        test_data_cytof,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        # collate_fn=collator,
        drop_last=False,
        num_workers=nw,
        pin_memory=True,
    )

    # Create model
    enc_config_model = HyenaConfig(
        d_model=config["enc_dim"],  # embedding 的维度
        emb_dim=config["position_emb_dim"],  # 必须是奇数且大于3，位置编码相关
        max_seq_len=config["enc_vocab_len"],  # 2000
        vocab_len=config["enc_vocab_len"],  # 19202
        n_layer=config["enc_depth"],
        output_hidden_states=False,
    )
    dec_config_model = HyenaConfig(
        d_model=config["dec_dim"],  # embedding 的维度
        emb_dim=config["position_emb_dim"],  # 必须是奇数且大于3，位置编码相关
        max_seq_len=config["dec_vocab_len"],  # 2000
        vocab_len=config["dec_vocab_len"],  # 19202
        n_layer=config["dec_depth"],
        output_hidden_states=False,
    )
    model = scTrans(enc_config_model, dec_config_model, mask_prob=config["mask_prob"], lr=config["lr"], emb_dropout=0.0)

    wandb_logger = TensorBoardLogger(
        save_dir="./10M/",
        name=args["wandb_name"]
    )

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss_epoch',
        filename='10M-{epoch:02d}-{valid_loss_epoch:.6f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        # precision=16,
        # accelerator='gpu', devices=[2,3],
        # accelerator=None,
        strategy="ddp_find_unused_parameters_true", accelerator="gpu", devices=config["device"],
        logger=wandb_logger,
        max_epochs=config["epochs"],
        val_check_interval=1.0,
        default_root_dir=weights_dir,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    torch.distributed.destroy_process_group()
    if trainer.global_rank == 0:
        trainer = pl.Trainer(accelerator='gpu', devices=[1], logger=wandb_logger)
        trainer.test(model, test_dataloader_citeseq)


if __name__ == "__main__":
    args = parser_args()
    main(args)
