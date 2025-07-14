import torch
from math import ceil
from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F
from model.modeling_hyena import HeynaModel
from model.tokenizer import (
    mask_data,
    data_tokenizer_padding,
    hierarchical_bayesian_downsampling,
)
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExpander(nn.Module):
    def __init__(self, p, d):
        super(FeatureExpander, self).__init__()
        self.p = p
        self.d = d
        self.linears = nn.ModuleList([nn.Linear(1, d) for _ in range(p)])

    def forward(self, x):
        expanded_features = [
            self.linears[i](x[:, i: i + 1]).unsqueeze(1) for i in range(self.p)
        ]
        output = torch.cat(expanded_features, dim=1)
        return output


class AutoDiscretizationBlock(nn.Module):
    def __init__(self, embedding_dim, num_tokens=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens

        # The weight vector w1 is a learnable parameter
        self.linear1 = nn.Linear(1, num_tokens)

        # The weight vector w2 should project to embedding_dim
        self.linear2 = nn.Linear(num_tokens, num_tokens)

        # Scaling factor alpha
        self.alpha = nn.Parameter(torch.rand(1))

        # Random lookup table
        self.EXP_lookup = nn.Parameter(torch.randn(num_tokens, embedding_dim))

    def forward(self, V):
        # Transform V with the linear layer and apply LeakyReLU
        v1 = self.linear1(V)
        v2 = F.leaky_relu(v1)

        # Project v2 with the linear layer and apply the scaling mixture factor
        v3 = self.linear2(v2) + self.alpha * v2

        # Normalize the weights using softmax across the num_tokens dimension
        v4 = F.softmax(v3, dim=-1)

        # Multiply the lookup table with v4 to get the final output of shape [c, n, embedding_dim]
        output = torch.matmul(v4, self.EXP_lookup)

        return output


class scHeyna(nn.Module):
    def __init__(self, config, emb_dropout=0.0, tie_embed=False):
        super().__init__()
        self.to_vector = nn.Linear(1, config.d_model)
        self.transformer = HeynaModel(config)
        self.to_out = nn.Linear(config.d_model, 1) if not tie_embed else None
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, return_encodings=False):
        b, n = x.shape[0], x.shape[1]
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=2)
            x = self.to_vector(x)

        x = self.dropout(x)
        x = self.transformer(inputs_embeds=x)
        if return_encodings:
            return x
        return torch.squeeze(self.to_out(x))


class MLPTranslator(nn.Module):
    """
    Class description: translator from RNA to protein
    fully connected layer with adjustable number of layers and variable dropout for each layer

    """

    # ----- Define all layers -----#
    def __init__(
            self,
            num_fc_input,
            num_output_nodes,
            num_fc_layers,
            initial_dropout,
            act=nn.ReLU(),
            **kwargs
    ):
        super(MLPTranslator, self).__init__(**kwargs)
        fc_d = pow(
            num_fc_input / num_output_nodes, 1 / num_fc_layers
        )  # reduce factor of fc layer dimension
        # --- Fully connected layers ---#
        self.num_fc_layers = num_fc_layers
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input / fc_d)))
            self.dropout0 = nn.Dropout(initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input / fc_d)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1, num_fc_layers - 1):
                    tmp_input = int(ceil(num_fc_input / fc_d ** i))
                    tmp_output = int(ceil(num_fc_input / fc_d ** (i + 1)))
                    exec("self.fc{} = nn.Linear(tmp_input, tmp_output)".format(i))
                    if (
                            i < ceil(num_fc_layers / 2)
                            and 1.1 ** (i + 1) * initial_dropout < 1
                    ):
                        exec(
                            "self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)".format(
                                i
                            )
                        )
                    elif (
                            i >= ceil(num_fc_layers / 2)
                            and 1.1 ** (num_fc_layers - 1 - i) * initial_dropout < 1
                    ):
                        exec(
                            "self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)".format(
                                i
                            )
                        )
                    else:
                        exec("self.dropout{} = nn.Dropout(initial_dropout)".format(i))
                # the last fc layer
                exec(
                    "self.fc{} = nn.Linear(tmp_output, num_output_nodes)".format(i + 1)
                )

        # --- Activation function ---#
        self.act = act

    # ----- Forward -----#
    def forward(self, x):
        # x size:  [batch size, feature_dim]

        if self.num_fc_layers == 1:
            outputs = self.fc0(x)
        else:
            # the first fc layer
            outputs = self.act(self.dropout0(self.fc0(x)))
            if self.num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                outputs = self.fc1(outputs)
            else:
                # the middle fc layer
                for i in range(1, self.num_fc_layers - 1):
                    outputs = eval(
                        "self.act(self.dropout{}(self.fc{}(outputs)))".format(i, i)
                    )
                # the last fc layer
                outputs = eval("self.fc{}(outputs)".format(i + 1))

        return outputs


class scTrans(pl.LightningModule):
    def __init__(
            self,
            enc_ret_config,
            dec_ret_config,
            encoder_ckpt_path,
            decoder_ckpt_path,
            lr=1e-4,
            emb_dropout=0.0,
            freeze_epochs=3,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = scHeyna(enc_ret_config, emb_dropout=emb_dropout, tie_embed=True)
        self.decoder = scHeyna(dec_ret_config, emb_dropout=emb_dropout)
        self.translator = MLPTranslator(
            enc_ret_config.max_seq_len, dec_ret_config.max_seq_len, 2, 0.1
        )

        self.load_encoder_decoder(encoder_ckpt_path, decoder_ckpt_path)

        self.cos_gene = nn.CosineSimilarity(dim=0, eps=1e-8)
        self.cos_cell = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []
        self.test_emb_list = []
        self.test_emb_mlp_list = []
        self.test_pred = []
        self.recon_list = []
        self.save_hyperparameters()

    def load_encoder_decoder(self, encoder_ckpt_path, decoder_ckpt_path):
        encoder_checkpoint = torch.load(encoder_ckpt_path)["state_dict"]
        encoder_state_dict = {
            k[len("encoder."):]: v
            for k, v in encoder_checkpoint.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state_dict, strict=False)

        decoder_checkpoint = torch.load(decoder_ckpt_path)["state_dict"]
        decoder_state_dict = {
            k[len("decoder."):]: v
            for k, v in decoder_checkpoint.items()
            if k.startswith("decoder.")
        }
        self.decoder.load_state_dict(decoder_state_dict, strict=False)

    def forward(self, seq_in):
        encodings = self.encoder(
            seq_in, return_encodings=True
        )  # batch_size, input_seq_lenth, dim
        seq_out = (
            self.translator(encodings.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )
        return encodings, seq_out, self.decoder(seq_out)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def recon_loss(self, data, pred_data, mask_idx=None):
        if mask_idx is None:
            return nn.functional.mse_loss(pred_data, data, reduction="mean")
        else:
            loss1 = self.masked_recon_loss(pred_data, data, mask_idx)
            return loss1

    def masked_recon_loss(self, data, pred_data, mask_idx=None):
        masked_squared_error = ((data - pred_data) ** 2) * mask_idx
        summed_error_per_row = masked_squared_error.sum(dim=1)
        mask_count_per_row = mask_idx.sum(dim=1).float()
        mask_count_per_row[mask_count_per_row == 0] = 1
        average_error_per_row = summed_error_per_row / mask_count_per_row
        recon_loss = average_error_per_row.mean()
        return recon_loss

    def mask_data_with_condition(self, tensor, condition_matrix, mask_probability=0.4):
        mask_non_zero = torch.rand_like(tensor) < mask_probability
        mask_zero = torch.rand_like(tensor) < (mask_probability / 10.0)
        mask_matrix = torch.where(tensor != 0, mask_non_zero, mask_zero)
        mask_matrix = torch.where(
            condition_matrix,
            torch.zeros_like(mask_matrix, dtype=torch.bool),
            mask_matrix,
        )

        return mask_matrix

    def training_step(self, batch, batch_idx):
        x, y, mask_idx = batch[0], batch[1], batch[2]
        embed, embed_mlp, recon = self(x)
        recon_loss = self.recon_loss(y, recon, mask_idx)

        corr_gene = self.cos_gene(y * mask_idx, recon * mask_idx).mean()
        corr_cell = self.cos_cell(y * mask_idx, recon * mask_idx).mean()

        metrics = {
            "train/loss": recon_loss,
            "train/corr_gene": corr_gene,
            "train/corr_cell": corr_cell,
        }
        for key, value in metrics.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return recon_loss

    def validation_step(self, batch, batch_idx):
        x, y, mask_idx = batch[0], batch[1], batch[2]

        embed, embed_mlp, recon = self(x)
        recon_loss = self.recon_loss(y, recon, mask_idx)

        corr_gene = self.cos_gene(y * mask_idx, recon * mask_idx).mean()
        corr_cell = self.cos_cell(y * mask_idx, recon * mask_idx).mean()

        metrics = {
            "valid/loss": recon_loss,
            "valid/corr_gene": corr_gene,
            "valid/corr_cell": corr_cell,
        }
        for key, value in metrics.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return {"val_loss": recon_loss}

    def test_step(self, batch, batch_idx):
        x, y, self.mask_idx = batch[0], batch[1], batch[2]

        self.embed, self.embed_mlp, self.recon = self(x)
        recon_loss = self.recon_loss(y, self.recon, self.mask_idx)

        corr_gene = self.cos_gene(y * self.mask_idx, self.recon * self.mask_idx).mean()
        corr_cell = self.cos_cell(y * self.mask_idx, self.recon * self.mask_idx).mean()

        metrics = {
            "test/loss": recon_loss,
            "test/corr_gene": corr_gene,
            "test/corr_cell": corr_cell,
        }

        for key, value in metrics.items():
            self.log(
                key,
                value,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        self.test_emb_list.append(self.embed.detach().to("cpu").numpy())
        self.test_emb_mlp_list.append(self.embed_mlp.detach().to("cpu").numpy())
        self.recon_list.append(self.recon.detach().to("cpu").numpy())

        return {"test_loss": recon_loss}

    def on_test_epoch_end(self):
        test_emb_1 = np.concatenate(self.test_emb_list, axis=0)
        emb_mean = np.mean(test_emb_1, axis=1)
        for i in range(min(10, len(self.embed))):
            self.plot_gene_umap(self.embed[i].detach().to("cpu").numpy())
            self.plot_corr_gene(self.embed[i].detach().to("cpu").numpy())

        test_emb_mlp = np.concatenate(self.test_emb_mlp_list, axis=0)
        test_emb_mlp = test_emb_mlp[:, self.mask_idx[0, :].detach().to("cpu").numpy(), :]
        emb_mean = np.mean(test_emb_mlp, axis=1)
        for i in range(min(3, len(self.embed_mlp))):
            self.plot_gene_umap(
                self.embed_mlp[i][self.mask_idx[0, :].detach().to("cpu").numpy(), :].detach().to("cpu").numpy(),
                name="Umap-Protein")

        self.test_emb_list = []
        self.test_pred = []

    def cluster(self, embed, reso=0.1):
        adata = sc.AnnData(embed)
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, resolution=reso, random_state=0)
        adata.obs["pred"] = adata.obs["louvain"]
        return adata

    def plot_gene_umap(self, emb, reso=0.5, name="umap-gene"):
        adata = sc.AnnData(emb)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=reso, random_state=0)
        adata.obs["label"] = adata.obs["louvain"].astype(int)
        adata.obs["label"] = adata.obs["label"].astype("category")
        fig = sc.pl.umap(adata, color=["label"], return_fig=True, show=False)
        # self.logger.experiment.log({name: wandb.Image(fig, caption=name)})
        self.logger.experiment.add_figure(name, fig)

    def plot_corr_gene(self, emb, gene_id=11150, name="Corr-Gene"):
        # CD4 correlation
        corr = cosine_similarity(emb)
        CD4_corr = corr[gene_id,]
        max_values_and_indices = sorted(
            enumerate(CD4_corr.tolist()), key=lambda x: x[1], reverse=True
        )[:10]
        min_values_and_indices = sorted(
            enumerate(CD4_corr.tolist()), key=lambda x: x[1]
        )[:10]
        max_indices, max_values = zip(*max_values_and_indices)
        min_indices, min_values = zip(*min_values_and_indices)
        gene_ann = pd.read_csv(
            "./gene_order_ann.csv", index_col=0
        )
        max_corr = gene_ann.iloc[list(max_indices),]["symbol"].values
        min_corr = gene_ann.iloc[list(min_indices),]["symbol"].values

        CD4_corr = CD4_corr.reshape(-1, 1)
        plt.figure(figsize=(6, 6))
        fig = sns.clustermap(CD4_corr, col_cluster=False, row_cluster=True)
        print(pd.DataFrame({"max_corr_gene": max_corr, "min_corr_gene": min_corr}))
        self.logger.experiment.add_figure(name, fig.fig)
        self.logger.experiment.add_text("Max corr", ', '.join(max_corr))
        self.logger.experiment.add_text("Min corr", ', '.join(min_corr))
