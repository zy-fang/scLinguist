import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.data_loader_pretrain import scRNADataset
import argparse
from model.model import scTrans
from model.configuration_hyena import HyenaConfig

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
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
        default="/public/home/hpc214701031/project/scTrans/data/RNA/cell_gene",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/media/asus/data16t/fangzy/result/scTrans/demo3_new/result",
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
        "--device", type=int, default=-1, help="device = 'cpu' or '0' or '0,1,2,3'"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--mask_prob", type=float, default=0.6)
    parser.add_argument("--hyena_dropout", type=float, default=0.0)
    parser.add_argument(
        "--enc_vocab_len", type=int, default=19202, help="vocab length of encoder"
    )
    parser.add_argument(
        "--dec_vocab_len", type=int, default=19202, help="vocab length of decoder"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2000, help="sequence length"
    )
    parser.add_argument(
        "--enc_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--dec_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=128, help="latent dimension of each token"
    )
    parser.add_argument(
        "--dec_dim", type=int, default=128, help="latent dimension of each token"
    )
    parser.add_argument(
        "--position_emb_dim",
        type=int,
        default=5,
        help="dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument("--wandb_name", type=str, default="test", help="wanbd name")

    args = parser.parse_args()
    config = vars(args)
    return config


def main(config):
    print(config)
    pl.seed_everything(config["seed"])

    # Create data
    all_data = spRNADataset(
        data_dir='/public/home/hpc214701031/project/scTrans/data/RNA/3m_cell_databanks_log',
        cache_dir="/public/home/hpc214701031/project/scTrans/data/RNA/3m_cell_cache")

    n_train = int(len(all_data) * 0.9)
    train_data, valid_data = torch.utils.data.random_split(
        all_data, [n_train, len(all_data) - n_train]
    )
    test_data = scRNADataset(
        data_dir="/public/home/hpc214701031/project/scTrans/data/RNA/test_data/PBMC_3.h5ad",
        gene_order_path="/public/home/hpc214701031/project/scTrans/data/RNA/gtf/gene_order.csv",
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
        shuffle=True,
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
        d_model=config["enc_dim"], 
        emb_dim=config["position_emb_dim"],  
        max_seq_len=config["enc_vocab_len"],
        vocab_len = config["enc_vocab_len"],
        n_layer=config["enc_depth"],
        output_hidden_states=False,
    )
    dec_config_model = HyenaConfig(
        d_model=config["dec_dim"],  
        emb_dim=config["position_emb_dim"], 
        max_seq_len=config["enc_vocab_len"],
        vocab_len = config["enc_vocab_len"], 
        n_layer=config["enc_depth"],
        output_hidden_states=False,
    )
    if args["checkpoint_path"] is None:
        model = scTrans(enc_config_model, dec_config_model, mask_prob=config["mask_prob"], lr=config["lr"])
    else:
        checkpoint_path=args["checkpoint_path"]
        print(checkpoint_path)
        model = scTrans.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=torch.device('cpu'))
    wandb_logger = TensorBoardLogger(
        save_dir="/public/home/hpc214701031/project/scTrans/result/RNA/demo3_new/",
        name=args["wandb_name"]
    )
    
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='valid/loss_epoch',
        filename='3M-{epoch:02d}-{val_loss:.6f}',
        save_top_k=-1,
        every_n_epochs=1,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true", accelerator="gpu",
        logger=wandb_logger,
        max_epochs=config["epochs"],
        val_check_interval=1.0,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        trainer = pl.Trainer(accelerator='gpu', devices=[0], logger=wandb_logger)
        trainer.test(model, test_dataloader)


if __name__ == "__main__":
    args = parser_args()
    main(args)
