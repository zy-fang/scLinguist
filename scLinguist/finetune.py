import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
import math
# import wandb
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
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.data_loader_finetune import scMultiDataset, spMultiDataset
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
from model.model_finetune import scTrans
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
        default="./result",
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
        "--device", type=str, default="", help="device = 'cpu' or '0' or '0,1,2,3'"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--mask_prob", type=float, default=0.6)
    parser.add_argument("--hyena_dropout", type=float, default=0.0)
    parser.add_argument(
        "--enc_vocab_len", type=int, default=19202, help="vocab length of encoder"
    )
    parser.add_argument(
        "--dec_vocab_len", type=int, default=6427, help="vocab length of decoder"
    )
    parser.add_argument("--seq_len", type=int, default=2000, help="sequence length")
    parser.add_argument(
        "--enc_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--dec_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=128, help="latend dimension of each token"
    )
    parser.add_argument(
        "--dec_dim", type=int, default=128, help="latend dimension of each token"
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

    # Create data
    all_data = spMultiDataset(
        data_dir_1="./3Mfinetune_RNA/all_X/",
        data_dir_2="./3Mfinetune_ADT/all_X/",
        mask_dir="./3Mfinetune_ADT_mask/"
    )
    n_train = int(len(all_data) * 0.9)
    train_data, valid_data = torch.utils.data.random_split(
        all_data, [n_train, len(all_data) - n_train]
    )
    test_data = scMultiDataset(
        # data_dir_1="./data/RNA_protein/test_data/dataset2_RNA.h5ad",
        # data_dir_2="./data/RNA_protein/test_data/dataset2_ADT_6427.h5ad",
        data_dir_1="./test_PR/dataset2_RNA.h5ad",
        data_dir_2="./test_PR/dataset2_full.h5ad",
    )

    nw = 8
    print("----------nw", nw)
    train_dataloader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
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
    model = scTrans(
        enc_config_model,
        dec_config_model,
        encoder_ckpt_path="../pretrained_model/3MRNA2025.ckpt",
        decoder_ckpt_path="../pretrained_model/16000/3M/1layer3M-notech-128/version_0/checkpoints/last.ckpt",
        lr=config["lr"],
    )
    wandb_logger = TensorBoardLogger(
        save_dir="./wandb",
        name=args["wandb_name"]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/loss_epoch",
        filename="300K-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        # accelerator="gpu", devices=[0],
        # accelerator=None,
        strategy="ddp_find_unused_parameters_true", accelerator="gpu",
        logger=wandb_logger,
        max_epochs=config["epochs"],
        val_check_interval=1.0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        trainer = pl.Trainer(accelerator='gpu', devices=[0], logger=wandb_logger)
        trainer.test(model, test_dataloader)


if __name__ == "__main__":
    args = parser_args()
    main(args)
