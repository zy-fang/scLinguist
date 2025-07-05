import torch
from math import ceil
from typing import Optional
import wandb
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn as nn, relu

import pytorch_lightning as pl
import torch.nn.functional as F

# from data_loader import normalization
from model.modeling_hyena import HeynaModel
from model.tokenizer_norm import mask_data, data_tokenizer_padding, hierarchical_bayesian_downsampling, maskorigin, masktest, maskorigin_new
from sklearn.metrics.pairwise import cosine_similarity

def max_min_normalization(data):
    High = 1e6
    Low = 0
    scale = High - Low

    min_vals = data.min(axis=1).toarray().ravel()
    max_vals = data.max(axis=1).toarray().ravel()

    nonzero_rows, nonzero_cols = data.nonzero()

    data.data = data.data.astype(float)
    data.data /= max_vals[nonzero_rows] - min_vals[nonzero_rows]
    data.data *= scale
    data.data += Low

    return data

class scHeyna_dec(nn.Module):
    def __init__(self, config, emb_dropout=0.0, tie_embed=False):
        super().__init__()
        self.to_vector = nn.Linear(1, config.d_model)
        self.to_vector_tech = nn.Linear(1, config.d_model)
        self.transformer = HeynaModel(config)
        self.to_out = nn.Linear(config.d_model, 1) if not tie_embed else None
        self.dropout = nn.Dropout(emb_dropout)
        # self.to_bin = nn.Linear(config.d_model, 1) if not tie_embed else None
        # self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, inputs, return_encodings=False):
        # print(inputs.shape)
        x = inputs[:, :, :]
        # tech = torch.unsqueeze(tech, dim=1)
        # print(tech)
        b, n = x.shape[0], x.shape[1]
        # if len(tech.shape) < 3:
        #     tech = torch.unsqueeze(tech, dim=2)
        #     tech = self.to_vector_tech(tech)
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=2)
            x = self.to_vector(x)
            # id = torch.arange(0, n).expand((b,n)).to(x.device)
            # pos = self.pos_emb(id)
            # x += pos
        # x = torch.cat((x, tech), dim=1)
        # print(x.shape)
        x = self.dropout(x)
        x = self.transformer(inputs_embeds=x)
        if return_encodings:
            return x
        # return torch.squeeze(self.to_out(x)), torch.squeeze(self.to_bin(x))
        return torch.squeeze(self.to_out(x))

class scHeyna_enc(nn.Module):
    def __init__(self, config, emb_dropout=0.0, tie_embed=False):
        super().__init__()
        self.to_vector = nn.Linear(1, config.d_model)
        self.to_vector_tech = nn.Linear(1, config.d_model)
        self.transformer = HeynaModel(config)
        self.to_out = nn.Linear(config.d_model, 1) if not tie_embed else None
        self.dropout = nn.Dropout(emb_dropout)
        # self.to_bin = nn.Linear(config.d_model, 1) if not tie_embed else None
        # self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, inputs, return_encodings=False):
        x, tech = inputs
        # tech = torch.unsqueeze(tech, dim=1)
        # print(tech)
        b, n = x.shape[0], x.shape[1]
        # if len(tech.shape) < 3:
        #     tech = torch.unsqueeze(tech, dim=2)
        #     tech = self.to_vector_tech(tech)
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=2)
            x = self.to_vector(x)
            # id = torch.arange(0, n).expand((b,n)).to(x.device)
            # pos = self.pos_emb(id)
            # x += pos
        # x = torch.cat((x, tech), dim=1)
        # print(x.shape)
        x = self.dropout(x)
        x = self.transformer(inputs_embeds=x)
        if return_encodings:
            return x
        # return torch.squeeze(self.to_out(x)), torch.squeeze(self.to_bin(x))
        return torch.squeeze(self.to_out(x))



class MLPTranslator(nn.Module):
    """
    Class description: translator from RNA to protein
    fully connected layer with adjustable number of layers and variable dropout for each layer

    """
    #----- Define all layers -----#
    def __init__(self, num_fc_input, num_output_nodes, num_fc_layers, initial_dropout, act = nn.ReLU(), **kwargs):
        super(MLPTranslator, self).__init__(**kwargs)
        fc_d = pow(num_fc_input/num_output_nodes,1/num_fc_layers) # reduce factor of fc layer dimension
        #--- Fully connected layers ---#
        self.num_fc_layers = num_fc_layers
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input/fc_d)))
            self.dropout0 = nn.Dropout(initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input/fc_d)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1,num_fc_layers-1):
                    tmp_input = int(ceil(num_fc_input/fc_d**i))
                    tmp_output = int(ceil(num_fc_input/fc_d**(i+1)))
                    exec('self.fc{} = nn.Linear(tmp_input, tmp_output)'.format(i))
                    if i < ceil(num_fc_layers/2) and 1.1**(i+1)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)'.format(i))
                    elif i >= ceil(num_fc_layers/2) and 1.1**(num_fc_layers-1-i)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)'.format(i))
                    else:
                        exec('self.dropout{} = nn.Dropout(initial_dropout)'.format(i))
                # the last fc layer
                exec('self.fc{} = nn.Linear(tmp_output, num_output_nodes)'.format(i+1))

        #--- Activation function ---#
        self.act = act

    #----- Forward -----#
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
                for i in range(1,self.num_fc_layers-1):
                    outputs = eval('self.act(self.dropout{}(self.fc{}(outputs)))'.format(i,i))
                # the last fc layer
                outputs = eval('self.fc{}(outputs)'.format(i+1))

        return outputs


class scTrans(pl.LightningModule):
    def __init__(self, enc_ret_config, dec_ret_config, mask_prob=0.3, lr=1e-4, emb_dropout=0.0):
        super().__init__()
        self.mask_prob = mask_prob
        self.lr = lr
        self.encoder = scHeyna_enc(
            enc_ret_config, emb_dropout=emb_dropout, tie_embed=True
        )
        self.decoder = scHeyna_dec(
            dec_ret_config, emb_dropout=emb_dropout
        )
        self.translator = MLPTranslator(enc_ret_config.max_seq_len, dec_ret_config.max_seq_len, 2, 0.5)
        self.cos_gene = nn.CosineSimilarity(dim=0, eps=1e-8)
        self.cos_cell = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []
        self.test_emb_list = []
        self.test_pred = []
        self.save_hyperparameters()

    # def forward(self, seq_in):
    #     encodings = self.encoder(
    #         seq_in, return_encodings=True
    #     )  # batch_size, input_seq_lenth, dim
    #     seq_out = self.translator(encodings.transpose(1,2).contiguous()).transpose(1,2).contiguous()
    #     return encodings, self.decoder(seq_out)

    def forward(self, seq_in):
        # nomlp
        encodings = self.encoder(
            seq_in, return_encodings=True
        )  # batch_size, input_seq_lenth, dim
        # seq_out = self.translator(encodings.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        return encodings, self.decoder(encodings)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def recon_loss(self, batch, pred_data, mask_idx=None):
        data, _ = batch
        if mask_idx is None:
            return nn.functional.mse_loss(pred_data, data, reduction='mean')
        else:
            # masked_data_weight = 1
            # w_nums = mask_idx * masked_data_weight + (~mask_idx) * (1 - masked_data_weight)
            # recon_loss =  torch.mul(w_nums, nn.functional.mse_loss(pred_data, data, reduction='none')).mean()
            masked_squared_error = ((data - pred_data) ** 2) * mask_idx
            summed_error_per_row = masked_squared_error.sum(dim=1)
            mask_count_per_row = mask_idx.sum(dim=1).float()
            mask_count_per_row[mask_count_per_row == 0] = 1
            average_error_per_row = summed_error_per_row / mask_count_per_row
            recon_loss = average_error_per_row.mean()
            return recon_loss


    def training_step(self, batch, batch_idx):
        # print(batch_idx, batch)
        # mask_x, mask_idx = maskorigin(batch, self.mask_prob)
        # print(batch[0][batch[1].type(torch.bool)])
        a, b, tech = batch
        _batch = (a, b)
        # mask_x, mask_idx, mask_final = maskorigin_new(_batch, self.mask_prob)
        mask_x, mask_idx, mask_final = mask_data(_batch, self.mask_prob)
        # print(list(batch[0][0]))
        if torch.isnan(_batch[0]).any():
            print("!!!!")
            print(list(_batch[0][0]))
        if torch.isnan(mask_x).any():
            print("!!!!")
            print(list(mask_x[0]))
        #     _batch[0][torch.isnan(_batch[0])] = torch.rand(
        #         _batch[0][torch.isnan(_batch[0])].shape).to(_batch[0].device)
        embed, recon = self((mask_x, tech))
        # print(list(recon[mask_final.type(torch.bool)]))
        # print(list(mask_x[mask_final.type(torch.bool)]))
        # print(list(a[mask_final.type(torch.bool)]))
        # print(batch[0][mask_idx.type(torch.bool)], recon[mask_idx.type(torch.bool)])
        # if torch.isnan(recon).any():
        #     print("1111")
        #     print(list(recon[0]))
        #     recon[torch.isnan(recon)] = torch.rand(
        #         recon[torch.isnan(recon)].shape).to(recon.device)
        recon_loss = self.recon_loss(_batch, recon, mask_final)
        # recon_loss2 = self.recon_loss(_batch, recon, mask_final) * 0.5
        # recon_loss = recon_loss1
        # print((_batch[0] * mask_final)[0])
        # print((recon * mask_final)[0])
        corr_gene = self.cos_gene(_batch[0] * mask_final, recon * mask_final).mean()
        corr_cell = self.cos_cell(_batch[0] * mask_final, recon * mask_final).mean()
        # recon_loss = 1 - corr_cell
        # print(batch[0][mask_idx.type(torch.bool)])
        # print(recon[mask_idx.type(torch.bool)])
        # print(corr_cell)
        metrics = {
            "train/loss": recon_loss,
            # "train/loss1": recon_loss1,
            # "train/loss2": recon_loss2,
            "train/corr_gene": corr_gene,
            "train/corr_cell": corr_cell
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
        # mask_x, mask_idx = maskorigin(batch, self.mask_prob)
        a, b, tech = batch
        _batch = (a, b)
        # mask_x, mask_idx, mask_final = maskorigin_new(_batch, self.mask_prob)
        mask_x, mask_idx, mask_final = mask_data(_batch, self.mask_prob)

        # if torch.isnan(_batch[0]).any():
        #     print("!!!!")
        #     print(list(_batch[0][0]))
        #     _batch[0][torch.isnan(_batch[0])] = torch.rand(
        #         _batch[0][torch.isnan(_batch[0])].shape).to(_batch[0].device)
        # embed, recon = self(mask_x)
        embed, recon = self((mask_x, tech))
        # if torch.isnan(recon).any():
        #     print("1111")
        #     print(list(recon[0]))
        #     recon[torch.isnan(recon)] = torch.rand(
        #         recon[torch.isnan(recon)].shape).to(recon.device)
        recon_loss = self.recon_loss(_batch, recon, mask_final)
        # recon_loss2 = self.recon_loss(_batch, recon, mask_final) * 0.5
        # recon_loss = recon_loss1
        corr_gene = self.cos_gene(_batch[0]* mask_final, recon* mask_final).mean()
        corr_cell = self.cos_cell(_batch[0]* mask_final, recon* mask_final).mean()
        # recon_loss = 1 - corr_cell
        metrics = {
            "valid_loss": recon_loss,
            # "valid/loss1": recon_loss1,
            # "valid/loss2": recon_loss2,
            "valid/corr_gene": corr_gene,
            "valid/corr_cell": corr_cell
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
        # mask_data, mask_idx = mask_data(data, self.mask_prob)
        a, b, tech = batch
        _batch = (a, b)
        # mask_x, mask_idx, mask_final = maskorigin_new(_batch, self.mask_prob)
        mask_x, mask_idx, mask_final = mask_data(_batch, self.mask_prob)
        print((a*mask_final)[0][(a*mask_final)[0] != 0])

        # mask_x, mask_idx = masktest(batch, self.mask_prob)
        if torch.isnan(_batch[0]).any():
            print("!!!!")
            print(list(_batch[0][0]))
            _batch[0][torch.isnan(_batch[0])] = torch.rand(
                _batch[0][torch.isnan(_batch[0])].shape).to(_batch[0].device)
        # self.embed, recon = self(mask_x)
        self.embed, recon = self((mask_x, tech))
        print((recon*mask_final)[0][(recon*mask_final)[0] != 0])

        if torch.isnan(recon).any():
            print("1111")
            print(recon[0])
            recon[torch.isnan(recon)] = torch.rand(
                recon[torch.isnan(recon)].shape).to(recon.device)
        recon_loss = self.recon_loss(_batch, recon, mask_final)
        # recon_loss2 = self.recon_loss(_batch, recon, mask_final) * 0.5
        # recon_loss = recon_loss1 + recon_loss2
        # recon_loss = self.recon_loss(batch, recon)
        corr_gene = self.cos_gene(_batch[0]* mask_final, recon* mask_final).mean()
        corr_cell = self.cos_cell(_batch[0]* mask_final, recon* mask_final).mean()
        metrics = {
            "test/loss": recon_loss,
            "test/corr_gene": corr_gene,
            "test/corr_cell": corr_cell
        }

        for key, value in metrics.items():
            self.log(
                key,
                value,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        # mean = np.mean(self.embed_mask.detach().to("cpu").numpy(), axis=1)
        # if len(mean.shape) == 1:
        #     mean = mean.reshape([1, mean.shape[0]])
        # self.test_emb_list.append(mean)
        # print(self.embed.shape)
        self.test_emb_list.append(self.embed.detach().to("cpu").numpy())

        return {"test_loss": recon_loss}

    # def on_test_epoch_end(self):
    #     test_emb_1 = np.concatenate(self.test_emb_list, axis=0)
    #     emb_mean = np.mean(test_emb_1, axis=1)

    #     label = self.cluster(emb_mean, reso=0.3)
    #     self.plot_umap(label)

    #     for i in range(min(10, len(self.embed_mask))):
    #         self.plot_gene_umap(self.embed[i].detach().to("cpu").numpy())
    #         # self.plot_corr_gene(self.embed_mask[i].detach().to("cpu").numpy())
    #         # self.plot_corr_gene_last(self.embed_mask[i].detach().to("cpu").numpy(), gene_id=180)
    #     # self.plot_gene_umap(self.embed.mean().detach().to("cpu").numpy())
    #     # self.plot_corr_gene(self.embed_mask.mean().detach().to("cpu").numpy())
    #     # self.plot_corr_gene_last(self.embed_mask.mean().detach().to("cpu").numpy(), gene_id=187)
    #     self.test_emb_list = []
    #     self.test_pred = []

    def cluster(self, embed, reso=0.1):
        adata = sc.AnnData(embed)
        sc.pp.neighbors(adata, n_neighbors=20)
        sc.tl.louvain(adata, resolution=reso, random_state=0)
        return adata.obs["louvain"].astype(int).to_numpy()

    def plot_gene_umap(self, emb, reso=0.5, name=""):
        adata = sc.AnnData(emb)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=reso, random_state=0)
        adata.obs["label"] = adata.obs["louvain"].astype(int)
        adata.obs["label"] = adata.obs["label"].astype("category")
        fig = sc.pl.umap(adata, color=["label"], return_fig=True, show=False)
        # self.logger.experiment.log({"umap_gene": wandb.Image(fig, caption="umap_gene")})
        self.logger.experiment.add_figure("umap_pros"+name, fig)

    def plot_corr_gene(self, emb, gene_id=0):
        # CD4 correlation
        print(emb.shape)
        # emb = emb[:, :64]
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
        max_corr = list(max_indices)
        min_corr = list(min_indices)
        CD4_corr = CD4_corr.reshape(-1, 1)
        plt.figure(figsize=(6, 6))
        print(CD4_corr)
        fig = sns.clustermap(CD4_corr, col_cluster=False, row_cluster=True)
        # self.logger.experiment.log({"Corr_heat": wandb.Image(fig.fig, caption="corr_gene")})
        self.logger.experiment.add_figure("Corr_heat", fig.fig)
        table = wandb.Table(
            dataframe=pd.DataFrame(
                {"max_corr_gene": max_corr, "min_corr_gene": min_corr}
            )
        )
        print(pd.DataFrame(
            {"max_corr_gene": max_corr, "min_corr_gene": min_corr}
        ))
        # self.logger.experiment.add_text("Max corr", ', '.join(max_corr))
        # self.logger.experiment.add_text("Min corr", ', '.join(min_corr))
        # self.logger.experiment.log({"gene_correlations": table})
        #
        # fig = sns.heatmap(CD4_corr)
        # self.logger.experiment.log({"Corr_heat": wandb.Image(fig, caption="corr_gene")})
        # table = wandb.Table(dataframe=pd.DataFrame({"max_corr_gene": max_corr, "min_corr_gene": min_corr}))
        # self.logger.experiment.log({"gene_correlations": table})

    def plot_corr_gene_last(self, emb, gene_id=0):
        # CD4 correlation
        print(emb.shape)
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
        max_corr = list(max_indices)
        min_corr = list(min_indices)
        CD4_corr = CD4_corr.reshape(-1, 1)
        plt.figure(figsize=(6, 6))
        fig = sns.clustermap(CD4_corr, col_cluster=False, row_cluster=True)
        # self.logger.experiment.log({"Corr_heat": wandb.Image(fig.fig, caption="corr_gene")})
        self.logger.experiment.add_figure("Corr_heat", fig.fig)
        table = wandb.Table(
            dataframe=pd.DataFrame(
                {"max_corr_gene": max_corr, "min_corr_gene": min_corr}
            )
        )
        print(pd.DataFrame(
            {"max_corr_gene": max_corr, "min_corr_gene": min_corr}
        ))
        # self.logger.experiment.add_text("Max corr", ', '.join(max_corr))
        # self.logger.experiment.add_text("Min corr", ', '.join(min_corr))
        # self.logger.experiment.log({"gene_correlations": table})
        # fig = sns.heatmap(CD4_corr)
        # self.logger.experiment.log({"Corr_heat": wandb.Image(fig, caption="corr_gene_last")})
        # table = wandb.Table(dataframe=pd.DataFrame({"max_corr_gene_last": max_corr, "min_corr_gene_last": min_corr}))
        # self.logger.experiment.log({"gene_correlations_last": table})